import * as dotenv from 'dotenv'
import OpenAI from 'openai'
import jwt_decode from 'jwt-decode'
import fetch from 'node-fetch'
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
import * as fs from 'node:fs/promises'
import * as path from 'node:path'
import { generateMessageId } from '../utils/id-generator'

const MODEL_CONFIGS: Record<string, { supportTopP: boolean; defaultTemperature?: number }> = {
  'gpt-5-search-api': { supportTopP: false, defaultTemperature: 0.8 },
}

function isTextFile(filename: string): boolean {
  const ext = path.extname(filename).toLowerCase()
  const textExtensions = ['.txt', '.md', '.json', '.csv', '.js', '.ts', '.py', '.java', '.html', '.css', '.xml', '.yml', '.yaml', '.log']
  return textExtensions.includes(ext)
}

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
  isImageModel,
}: Pick<OpenAI.ChatCompletionCreateParams, 'temperature' | 'model' | 'top_p'> & {
  maxContextCount: number
  content: MessageContent
  abortSignal?: AbortSignal
  systemMessage?: string
  lastMessageId?: string
  isImageModel?: boolean
}) {
  const config = await getCacheConfig()
  const OPENAI_API_BASE_URL = isNotEmptyString(key.baseUrl) ? key.baseUrl : config.apiBaseUrl
  const openai = new OpenAI({
    baseURL: OPENAI_API_BASE_URL,
    apiKey: key.key,
    maxRetries: 0,
  })
  
  const modelConfig = MODEL_CONFIGS[model] || { supportTopP: true }
  const finalTemperature = modelConfig.defaultTemperature ?? temperature
  const shouldUseTopP = modelConfig.supportTopP

  const messages: OpenAI.ChatCompletionMessageParam[] = []
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId)
      break
    const message = await getMessageById(lastMessageId)
    if (!message)
      break
    
    // 即使是 OpenAI 调用，也确保 content 里没有超长 base64
    let safeContent = message.text
    if (typeof safeContent === 'string') {
       // 二次清洗，防止 getMessageById 漏网（虽然那里已经处理了）
       safeContent = safeContent.replace(/!\[.*?\]\(data:image\/.*?;base64,.*?\)/g, '[Image Data Removed]')
    }

    messages.push({
      role: message.role as any,
      content: safeContent,
    })
    lastMessageId = message.parentMessageId
  }
  if (systemMessage) {
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

  const enableStream = !isImageModel

  const options: OpenAI.ChatCompletionCreateParams = {
    model,
    stream: enableStream,
    stream_options: enableStream ? { include_usage: true } : undefined,
    messages,
  }
  options.temperature = finalTemperature
  if (shouldUseTopP) {
    options.top_p = top_p
  }

  // [调试日志]
  console.log(`[OpenAI] Request Model: ${model}, URL: ${OPENAI_API_BASE_URL}`)

  const apiResponse = await openai.chat.completions.create(options, {
    signal: abortSignal,
  })

  return apiResponse as AsyncIterable<OpenAI.ChatCompletionChunk>
}

const processThreads: { userId: string; abort: AbortController; messageId: string }[] = []

/**
 * 为 Gemini 构造多轮对话结构的 contents（仅文本，不带历史图片）
 * 结构遵循：
 * contents: [
 *   { role: 'user' | 'model', parts: [{ text: '...' }] },
 *   ...
 * ]
 */
async function buildGeminiContents(options: {
  lastMessageId?: string
  systemMessage?: string
  currentPromptText: string
  maxContextCount: number
}) {
  const { lastMessageId, systemMessage, currentPromptText, maxContextCount } = options

  type GeminiTurn = { role: 'user' | 'model'; text: string }

  const historyTurns: GeminiTurn[] = []

  let cursorId = lastMessageId
  for (let i = 0; i < maxContextCount; i++) {
    if (!cursorId)
      break
    const msg = await getMessageById(cursorId)
    if (!msg)
      break

    // 提取纯文本（忽略历史里的 image_url 等，以免上下文膨胀）
    let textStr = ''
    const raw = msg.text
    if (typeof raw === 'string') {
      textStr = raw
    }
    else if (Array.isArray(raw)) {
      for (const part of raw) {
        // OpenAI.ChatCompletionContentPart 里 type === 'text' 时才取文本
        if ((part as any).type === 'text' && (part as any).text)
          textStr += `${(part as any).text}\n`
      }
    }

    if (textStr) {
      // 进一步清洗掉可能残留的 base64 图片
      textStr = textStr.replace(/!\[.*?\]\(data:image\/.*?;base64,.*?\)/g, '[Image Data Removed]')
      // 把历史中 /uploads/xxx 的图标记为占位，避免模型对本地 URL 产生误解
      textStr = textStr.replace(/!\[.*?\]\(\/uploads\/.*?\)/g, '[Image]')
    }

    const role: 'user' | 'model'
      = msg.role === 'assistant'
        ? 'model'
        : 'user'

    historyTurns.push({ role, text: textStr })

    cursorId = msg.parentMessageId
  }

  // 从最早到最近的顺序
  historyTurns.reverse()

  const contents: any[] = []

  if (systemMessage) {
    contents.push({
      role: 'user',
      parts: [{ text: systemMessage }],
    })
  }

  for (const turn of historyTurns) {
    if (!turn.text?.trim())
      continue
    contents.push({
      role: turn.role,
      parts: [{ text: turn.text }],
    })
  }

  // 当前这一轮用户请求
  contents.push({
    role: 'user',
    parts: [{ text: currentPromptText }],
  })

  return contents
}

async function chatReplyProcess(options: RequestOptions): Promise<{ message: string; data: ChatResponse; status: string }> {
  const model = options.room.chatModel
  const key = await getRandomApiKey(options.user, model, options.room.accountId)
  const userId = options.user._id.toString()
  const maxContextCount = options.user.advanced.maxContextCount ?? 20
  const messageId = options.messageId
  if (key == null || key === undefined)
    throw new Error('没有对应的 apikeys 配置，请再试一次。')

  updateRoomChatModel(userId, options.room.roomId, model)

  const { message, uploadFileKeys, lastContext, process, systemMessage, temperature, top_p } = options
  let content: MessageContent = message
  let fileContext = ''
  const globalConfig = await getCacheConfig()
  const imageModelsStr = globalConfig.siteConfig.imageModels || ''
  const imageModelList = imageModelsStr.split(/[,，]/).map(s => s.trim()).filter(Boolean)
  
  const isImage = imageModelList.some(m => model.includes(m))
  // 只要是图片模型且名字包含 gemini 即可
  const isGeminiImageModel = isImage && model.includes('gemini')


  if (uploadFileKeys && uploadFileKeys.length > 0) {
    const textFiles = uploadFileKeys.filter(key => isTextFile(key))
    const imageFiles = uploadFileKeys.filter(key => !isTextFile(key))

    if (textFiles.length > 0) {
      for (const fileKey of textFiles) {
        try {
          const filePath = path.join('uploads', fileKey)
          const fileContent = await fs.readFile(filePath, 'utf-8')
          fileContext += `\n\n--- File Start: ${fileKey} ---\n${fileContent}\n--- File End ---\n`
        } catch (e) {
          console.error(`Error reading file ${fileKey}`, e)
        }
      }
    }

    const finalMessage = message + (fileContext ? `\n\nAttached Files Content:\n${fileContext}` : '')

    if (imageFiles.length > 0) {
      content = [
        {
          type: 'text',
          text: finalMessage,
        },
      ]
      for (const uploadFileKey of imageFiles) {
        content.push({
          type: 'image_url',
          image_url: {
            url: await convertImageUrl(uploadFileKey),
          },
        })
      }
    } else {
      content = finalMessage
    }
  }

  const abort = new AbortController()
  const customMessageId = generateMessageId()

  try {
    // ====== ① Gemini 图片模型（多轮结构，仅文本历史） ======
    if (isGeminiImageModel) {
      const GEMINI_BASE_URL = isNotEmptyString(key.baseUrl)
        ? key.baseUrl
        : 'https://generativelanguage.googleapis.com'

      const endpoint = `${GEMINI_BASE_URL.replace(/\/+$/, '')}/v1beta/models/gemini-3-pro-image:generateContent?key=${encodeURIComponent(key.key)}`

      // 当前这轮的 prompt 文本
      let promptText = ''
      if (typeof content === 'string') {
        promptText = content
      }
      else if (Array.isArray(content)) {
        for (const part of content) {
          if ((part as any).type === 'text' && (part as any).text)
            promptText += `${(part as any).text}\n`
        }
      }

      // 构造符合 Gemini 标准多轮结构的 contents
      const contents = await buildGeminiContents({
        lastMessageId: lastContext?.parentMessageId,
        systemMessage,
        currentPromptText: promptText,
        maxContextCount,
      })

      const body = { contents }

      processThreads.push({ userId, abort, messageId })

      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: abort.signal,
        body: JSON.stringify(body),
      })

      if (!resp.ok) {
        const errText = await resp.text()
        throw new Error(`Gemini API error: ${resp.status} ${errText}`)
      }

      const data: any = await resp.json()
      let text = ''
      
      const candidate = data.candidates?.[0]
      const parts = candidate?.content?.parts ?? []

      for (const part of parts) {
        if (part.text) text += part.text

        if (part.inlineData?.data) {
          const mime = part.inlineData.mimeType || 'image/png'
          const base64 = part.inlineData.data as string
          const buffer = Buffer.from(base64, 'base64')
          const ext = mime.split('/')[1] || 'png'
          const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
          const filePath = path.join('uploads', filename)
          await fs.writeFile(filePath, buffer)
          
          const prefix = text ? '\n\n' : ''
          text += `${prefix}![Generated Image](/uploads/${filename})`
        }
      }

      if (!text) text = JSON.stringify(data)

      const usageRes: any = data.usageMetadata
        ? {
            prompt_tokens: data.usageMetadata.promptTokenCount ?? 0,
            completion_tokens: data.usageMetadata.candidatesTokenCount ?? 0,
            total_tokens: data.usageMetadata.totalTokenCount ?? 0,
            estimated: true,
          }
        : undefined

      process?.({
        id: customMessageId,
        text,
        role: 'assistant',
        conversationId: lastContext.conversationId,
        parentMessageId: lastContext.parentMessageId,
        detail: undefined,
      })

      return sendResponse({
        type: 'Success',
        data: {
          object: 'chat.completion',
          choices: [{
            message: { role: 'assistant', content: text },
            finish_reason: 'stop',
            index: 0,
            logprobs: null,
          }],
          created: Date.now(),
          conversationId: lastContext.conversationId,
          model,
          text,
          id: customMessageId,
          detail: { usage: usageRes },
        },
      })
    }

    // ====== ② 其它模型 (OpenAI SDK) ======
    const api = await initApi(key, {
      model,
      maxContextCount,
      temperature,
      top_p,
      content,
      abortSignal: abort.signal,
      systemMessage,
      lastMessageId: lastContext.parentMessageId,
      isImageModel: isImage,
    })
    processThreads.push({ userId, abort, messageId })

    let text = ''
    let chatIdRes = customMessageId
    let modelRes = ''
    let usageRes: OpenAI.Completions.CompletionUsage

    if (isImage) {
      const response = api as any
      const choice = response.choices[0]
      let rawContent = choice.message?.content || ''
      modelRes = response.model
      usageRes = response.usage

      if (rawContent && !rawContent.startsWith('![') && (rawContent.startsWith('http') || rawContent.startsWith('data:image'))) {
         text = `![Generated Image](${rawContent})`
      } else {
         text = rawContent
      }

      process?.({
        id: customMessageId,
        text,
        role: choice.message.role || 'assistant',
        conversationId: lastContext.conversationId,
        parentMessageId: lastContext.parentMessageId,
        detail: {
            choices: [{ finish_reason: 'stop', index: 0, logprobs: null, message: choice.message }],
            created: response.created,
            id: response.id,
            model: response.model,
            object: 'chat.completion',
            usage: response.usage
        } as any 
      })

    } else {
      for await (const chunk of api as AsyncIterable<OpenAI.ChatCompletionChunk>) {
         text += chunk.choices[0]?.delta.content ?? ''
         chatIdRes = customMessageId
         modelRes = chunk.model
         usageRes = usageRes || chunk.usage
         process?.({
           ...chunk,
           id: customMessageId,
           text,
           role: chunk.choices[0]?.delta.role || 'assistant',
           conversationId: lastContext.conversationId,
           parentMessageId: lastContext.parentMessageId,
         })
      }
    }

    return sendResponse({
      type: 'Success',
      data: {
        object: 'chat.completion',
        choices: [{
          message: { role: 'assistant', content: text },
          finish_reason: 'stop',
          index: 0,
          logprobs: null,
        }],
        created: Date.now(),
        conversationId: lastContext.conversationId,
        model: modelRes,
        text,
        id: chatIdRes,
        detail: {
          usage: usageRes && { ...usageRes, estimated: false },
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
  if (index <= -1) return
  const messageId = processThreads[index].messageId
  processThreads[index].abort.abort()
  processThreads.splice(index, 1)
  return messageId
}

export function initAuditService(audit: AuditConfig) {
  if (!audit || !audit.options || !audit.options.apiKey || !audit.options.apiSecret) return
  const Service = textAuditServices[audit.provider]
  auditService = new Service(audit.options)
}

async function containsSensitiveWords(audit: AuditConfig, text: string): Promise<boolean> {
  if (audit.customizeEnabled && isNotEmptyString(audit.sensitiveWords)) {
    const textLower = text.toLowerCase()
    const notSafe = audit.sensitiveWords.split('\n').filter(d => textLower.includes(d.trim().toLowerCase())).length > 0
    if (notSafe) return true
  }
  if (audit.enabled) {
    if (!auditService) initAuditService(audit)
    return await auditService.containsSensitiveWords(text)
  }
  return false
}

async function chatConfig() {
  const config = await getOriginConfig() as ModelConfig
  return sendResponse<ModelConfig>({ type: 'Success', data: config })
}

// [重要修改] 在获取历史消息时，清洗掉可能存在的旧 Base64 数据
async function getMessageById(id: string): Promise<ChatMessage | undefined> {
  const isPrompt = id.startsWith('prompt_')
  const chatInfo = await getChatByMessageId(isPrompt ? id.substring(7) : id)

  if (chatInfo) {
    const parentMessageId = isPrompt
      ? chatInfo.options.parentMessageId
      : `prompt_${id}`

    if (chatInfo.status !== Status.Normal) {
      return parentMessageId ? getMessageById(parentMessageId) : undefined
    }
    else {
      if (isPrompt) {
        let promptText = chatInfo.prompt
        const allFileKeys = chatInfo.images || []
        const textFiles = allFileKeys.filter(k => isTextFile(k))
        const imageFiles = allFileKeys.filter(k => !isTextFile(k))

        if (textFiles.length > 0) {
          let fileContext = ''
          for (const fileKey of textFiles) {
            try {
              const filePath = path.join('uploads', fileKey)
              const fileContent = await fs.readFile(filePath, 'utf-8')
              fileContext += `\n\n--- Context File: ${fileKey} ---\n${fileContent}\n--- End File ---\n`
            } catch (e) {
              console.error(`Error reading history file ${fileKey}`, e)
            }
          }
          promptText += (fileContext ? `\n\n[Attached Files History]:\n${fileContext}` : '')
        }
        
        // 强制清理 Prompt 中的 Base64 脏数据
        if (promptText && typeof promptText === 'string') {
           promptText = promptText.replace(/!\[.*?\]\(data:image\/.*?;base64,.*?\)/g, '[Image Data Removed]')
        }

        let content: MessageContent = promptText

        if (imageFiles.length > 0) {
          content = [{ type: 'text', text: promptText }]
          for (const image of imageFiles) {
            content.push({
              type: 'image_url',
              image_url: { url: await convertImageUrl(image) },
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
        // [Fix] 清洗 Assistant 回复中的 Base64 图片
        // 如果历史记录里有旧的 base64 图片，这里会把它变成简短文字，从而避免后续请求 Token 爆炸
        let responseText = chatInfo.response || ''
        if (responseText.includes('data:image') && responseText.includes('base64')) {
            // 正则替换 Markdown 图片语法中包含 data:image 的部分
            responseText = responseText.replace(/!\[.*?\]\(data:image\/.*?;base64,.*?\)/g, '[Image History]')
        }
        
        return {
          id,
          conversationId: chatInfo.options.conversationId,
          parentMessageId,
          role: 'assistant',
          text: responseText,
        }
      }
    }
  }
  else { return undefined }
}

async function randomKeyConfig(keys: KeyConfig[]): Promise<KeyConfig | null> {
  if (keys.length <= 0) return null
  _lockedKeys.filter(d => d.lockedTime <= Date.now() - 1000 * 20).forEach(d => _lockedKeys.splice(_lockedKeys.indexOf(d), 1))
  let unsedKeys = keys.filter(d => _lockedKeys.filter(l => d.key === l.key).length <= 0)
  const start = Date.now()
  while (unsedKeys.length <= 0) {
    if (Date.now() - start > 3000) break
    await new Promise(resolve => setTimeout(resolve, 1000))
    unsedKeys = keys.filter(d => _lockedKeys.filter(l => d.key === l.key).length <= 0)
  }
  if (unsedKeys.length <= 0) return null
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
