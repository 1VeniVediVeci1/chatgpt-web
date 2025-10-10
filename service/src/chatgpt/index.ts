import * as dotenv from 'dotenv'
import OpenAI from 'openai'
import jwt_decode from 'jwt-decode'
import type { AuditConfig, KeyConfig, UserInfo } from '../storage/model'
import { Status } from '../storage/model'
import { convertImageUrl } from '../utils/image'
import type { TextAuditService } from '../utils/textAudit'
import { textAuditServices } from '../utils/textAudit'
import { getCacheApiKeys, getCacheConfig, getOriginConfig } from '../storage/config'
import { sendResponse } from '../utils'
import { hasAnyRole, isNotEmptyString } from '../utils/is'
import type { JWT, ModelConfig } from '../types'
import { getChatByMessageId, updateRoomChatModel } from '../storage/mongo'
import type { ChatMessage, ChatResponse, MessageContent, RequestOptions } from './types'

dotenv.config()

const ErrorCodeMessage: Record<string, string> = {
  401: '提供错误的API密钥',
  403: '服务器拒绝访问，请稍后再试',
  502: '错误的网关',
  503: '服务器繁忙，请稍后再试',
  504: '网关超时',
  500: '服务器繁忙，请稍后再试',
}

let auditService: TextAuditService
const _lockedKeys: { key: string; lockedTime: number }[] = []

export async function initApi(key: KeyConfig, {
  model,
  maxContextCount,
  temperature,
  top_p,
  abortSignal,
  content,
  systemMessage,
  lastMessageId,
}: Pick<OpenAI.ChatCompletionCreateParams, 'temperature' | 'model' | 'top_p'> & {
  maxContextCount: number
  content: MessageContent
  abortSignal?: AbortSignal
  systemMessage?: string
  lastMessageId?: string
}) {
  const config = await getCacheConfig()
  const OPENAI_API_BASE_URL = isNotEmptyString(key.baseUrl) ? key.baseUrl : config.apiBaseUrl
  const openai = new OpenAI({
    baseURL: OPENAI_API_BASE_URL,
    apiKey: key.key,
    maxRetries: 0,
  })
  const messages: OpenAI.ChatCompletionMessageParam[] = []
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId)
      break
    const message = await getMessageById(lastMessageId)
    if (!message)
      break
    messages.push({
      role: message.role as any,
      content: message.text,
    })
    lastMessageId = message.parentMessageId
  }
  // 判断模型是否为 o1
  const isO1Model = model.includes('o1')
  if (systemMessage && !isO1Model) {
    messages.push({
      role: 'system',
      content: systemMessage,
    })
  }
  messages.reverse()
  messages.push({
    role: 'user',
    content,
  })



  const options: OpenAI.ChatCompletionCreateParams = {
    model,
    top_p,
    stream: !isO1Model,  // 如果是 o1 模型，则禁用流式传输
    stream_options: isO1Model ? undefined : { include_usage: true },  // 只有在使用流式传输时才包含 stream_options
    messages,
  }

  if (!isO1Model) {
    options.temperature = temperature
  }

  // 根据是否使用流式传输返回不同的结果
  const apiResponse = await openai.chat.completions.create(options, {
    signal: abortSignal,
  })

  // 如果不使用流式传输，直接返回响应
  if (isO1Model) {
    return apiResponse as OpenAI.ChatCompletion // 如果不是流式传输，直接返回完整响应
  }

  // 否则，返回异步可迭代的流
  return apiResponse as AsyncIterable<OpenAI.ChatCompletionChunk> // 使用 AsyncIterable 代替 Stream
}

const processThreads: { userId: string; abort: AbortController; messageId: string }[] = []

async function chatReplyProcess(options: RequestOptions): Promise<{ message: string; data: ChatResponse; status: string }> {
  const model = options.room.chatModel
  const key = await getRandomApiKey(options.user, model, options.room.accountId)
  const userId = options.user._id.toString()
  const maxContextCount = options.user.advanced.maxContextCount ?? 20
  const messageId = options.messageId
  if (key == null || key === undefined)
    throw new Error('没有对应的 apikeys 配置，请再试一次。')

  // Add Chat Record
  updateRoomChatModel(userId, options.room.roomId, model)

  const { message, uploadFileKeys, lastContext, process, systemMessage, temperature, top_p } = options
  let content: MessageContent = message
  if (uploadFileKeys && uploadFileKeys.length > 0) {
    content = [
      {
        type: 'text',
        text: message,
      },
    ]
    for (const uploadFileKey of uploadFileKeys) {
      content.push({
        type: 'image_url',
        image_url: {
          url: await convertImageUrl(uploadFileKey),
        },
      })
    }
  }

  const abort = new AbortController()

  try {
    const api = await initApi(key, {
      model,
      maxContextCount,
      temperature,
      top_p,
      content,
      abortSignal: abort.signal,
      systemMessage,
      lastMessageId: lastContext.parentMessageId,
    })
    processThreads.push({ userId, abort, messageId })

    let text = ''
    let chatIdRes = null
    let modelRes = ''
    let usageRes: OpenAI.Completions.CompletionUsage
    // 🔥 确保 conversationId 始终存在
    const conversationId = lastContext.conversationId || `conv-${Date.now()}`
    // 如果是流式传输，使用 for await 迭代 response
    // 流式传输，逐块处理
    for await (const chunk of api as AsyncIterable<OpenAI.ChatCompletionChunk>) { // 使用 AsyncIterable
      text += chunk.choices[0]?.delta.content ?? ''
      // 只在第一次或 API 返回时赋值
      if (!chatIdRes && chunk.id) {
        chatIdRes = chunk.id
      }
      modelRes = chunk.model
      usageRes = usageRes || chunk.usage

      console.warn('[chunk]', chunk)
      process?.({
        ...chunk,
        text,
        role: chunk.choices[0]?.delta.role || 'assistant',
        conversationId, // 使用确保存在的 conversationId
        parentMessageId: lastContext.parentMessageId,
      })
    }

    // 如果流式传输后仍无 id，生成一个
    if (!chatIdRes) {
      chatIdRes = `msg-${Date.now()}`
    }
    return sendResponse({
      type: 'Success',
      data: {
        object: 'chat.completion',
        choices: [{
          message: {
            role: 'assistant',
            content: text,
          },
          finish_reason: 'stop',
          index: 0,
          logprobs: null,
        }],
        created: Date.now(),
        conversationId, // 使用确保存在的值
        model: modelRes,
        text,
        id: chatIdRes,
        detail: {
          usage: usageRes && {
            ...usageRes,
            estimated: false,
          },
        },
      },
    })
  }
  catch (error: any) {
    const code = error.statusCode
    if (code === 429 && (error.message.includes('Too Many Requests') || error.message.includes('Rate limit'))) {
      if (options.tryCount++ < 3) {
        _lockedKeys.push({ key: key.key, lockedTime: Date.now() })
        await new Promise(resolve => setTimeout(resolve, 2000))
        return await chatReplyProcess(options)
      }
    }
    globalThis.console.error(error)
    if (Reflect.has(ErrorCodeMessage, code))
      return sendResponse({ type: 'Fail', message: ErrorCodeMessage[code] })
    return sendResponse({ type: 'Fail', message: error.message ?? 'Please check the back-end console' })
  }
  finally {
    const index = processThreads.findIndex(d => d.userId === userId)
    if (index > -1)
      processThreads.splice(index, 1)
  }
}

export function abortChatProcess(userId: string) {
  const index = processThreads.findIndex(d => d.userId === userId)
  if (index <= -1)
    return
  const messageId = processThreads[index].messageId
  processThreads[index].abort.abort()
  processThreads.splice(index, 1)
  return messageId
}

export function initAuditService(audit: AuditConfig) {
  if (!audit || !audit.options || !audit.options.apiKey || !audit.options.apiSecret)
    return
  const Service = textAuditServices[audit.provider]
  auditService = new Service(audit.options)
}

async function containsSensitiveWords(audit: AuditConfig, text: string): Promise<boolean> {
  if (audit.customizeEnabled && isNotEmptyString(audit.sensitiveWords)) {
    const textLower = text.toLowerCase()
    const notSafe = audit.sensitiveWords.split('\n').filter(d => textLower.includes(d.trim().toLowerCase())).length > 0
    if (notSafe)
      return true
  }
  if (audit.enabled) {
    if (!auditService)
      initAuditService(audit)
    return await auditService.containsSensitiveWords(text)
  }
  return false
}

async function chatConfig() {
  const config = await getOriginConfig() as ModelConfig
  return sendResponse<ModelConfig>({
    type: 'Success',
    data: config,
  })
}

async function getMessageById(id: string): Promise<ChatMessage | undefined> {
  const isPrompt = id.startsWith('prompt_')
  const chatInfo = await getChatByMessageId(isPrompt ? id.substring(7) : id)

  if (chatInfo) {
    const parentMessageId = isPrompt
      ? chatInfo.options.parentMessageId
      : `prompt_${id}` // parent message is the prompt

    if (chatInfo.status !== Status.Normal) { // jumps over deleted messages
      return parentMessageId
        ? getMessageById(parentMessageId)
        : undefined
    }
    else {
      if (isPrompt) { // prompt
        let content: MessageContent = chatInfo.prompt
        if (chatInfo.images && chatInfo.images.length > 0) {
          content = [
            {
              type: 'text',
              text: chatInfo.prompt,
            },
          ]
          for (const image of chatInfo.images) {
            content.push({
              type: 'image_url',
              image_url: {
                url: await convertImageUrl(image),
              },
            })
          }
        }
        return {
          id,
          conversationId: chatInfo.options.conversationId,
          parentMessageId,
          role: 'user',
          text: content,
        }
      }
      else {
        return { // completion
          id,
          conversationId: chatInfo.options.conversationId,
          parentMessageId,
          role: 'assistant',
          text: chatInfo.response,
        }
      }
    }
  }
  else { return undefined }
}

async function randomKeyConfig(keys: KeyConfig[]): Promise<KeyConfig | null> {
  if (keys.length <= 0)
    return null
  // cleanup old locked keys
  _lockedKeys.filter(d => d.lockedTime <= Date.now() - 1000 * 20).forEach(d => _lockedKeys.splice(_lockedKeys.indexOf(d), 1))

  let unsedKeys = keys.filter(d => _lockedKeys.filter(l => d.key === l.key).length <= 0)
  const start = Date.now()
  while (unsedKeys.length <= 0) {
    if (Date.now() - start > 3000)
      break
    await new Promise(resolve => setTimeout(resolve, 1000))
    unsedKeys = keys.filter(d => _lockedKeys.filter(l => d.key === l.key).length <= 0)
  }
  if (unsedKeys.length <= 0)
    return null
  const thisKey = unsedKeys[Math.floor(Math.random() * unsedKeys.length)]
  return thisKey
}

async function getRandomApiKey(user: UserInfo, chatModel: string, accountId?: string): Promise<KeyConfig | undefined> {
  let keys = (await getCacheApiKeys()).filter(d => hasAnyRole(d.userRoles, user.roles))
    .filter(d => d.chatModels.includes(chatModel))
  if (accountId)
    keys = keys.filter(d => d.keyModel === 'ChatGPTUnofficialProxyAPI' && getAccountId(d.key) === accountId)

  return randomKeyConfig(keys)
}

function getAccountId(accessToken: string): string {
  try {
    const jwt = jwt_decode(accessToken) as JWT
    return jwt['https://api.openai.com/auth'].user_id
  }
  catch (error) {
    return ''
  }
}

export { chatReplyProcess, chatConfig, containsSensitiveWords }
