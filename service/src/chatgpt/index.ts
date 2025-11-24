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
import * as fs from 'node:fs/promises' // 引入 fs
import * as path from 'node:path'      // 引入 path
// 导入自定义 ID 生成器
import { generateMessageId } from '../utils/id-generator'

// 在文件开头添加模型配置
const MODEL_CONFIGS: Record<string, { supportTopP: boolean; defaultTemperature?: number }> = {
  'gpt-5-search-api': { supportTopP: false, defaultTemperature: 0.8 },
  // 其他特殊模型可以在这里添加
}

// 添加一个辅助函数来判断是否为文本文件
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
  //传入是否为图片模型的标记
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
  // 根据模型类型调整参数
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
    
    // --- 核心修改开始：自动将生成的 Base64 图片转为下一轮附件 ---
    let historyContent: any = message.text

    // 正则表达式：匹配 Markdown 图片语法中的 Base64 数据
    // 格式: ![描述](data:image/png;base64,...)
    const base64ImgRegex = /!\[(.*?)\]\((data:image\/.*?;base64,.*?)\)/

    // 检查历史记录是否包含 Base64 图片
    if (typeof historyContent === 'string' && base64ImgRegex.test(historyContent)) {
      const match = historyContent.match(base64ImgRegex)
      
      if (match && match[2]) {
        const altText = match[1] || 'Generated Image'
        const imageUrl = match[2] // 提取出 data:image... Base64字符串

        // 1. 清理文本：把巨大的 Base64 字符串替换成简短的占位符，防止 Text Token 爆炸
        const cleanText = historyContent.replace(base64ImgRegex, `[图片: ${altText}]`)

        // 2. 重构为多模态格式：将图片作为 image_url 附件发送给模型
        // 这样模型能看到上一轮生成的图，从而支持基于该图的修改指令
        historyContent = [
          {
            type: 'text',
            text: cleanText, 
          },
          {
            type: 'image_url',
            image_url: {
              url: imageUrl, // 这里作为真正的 Vision Input 发送
            },
          },
        ]
      }
    }
    // --- 核心修改结束 ---

    messages.push({
      role: message.role as any,
      content: historyContent,
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
  // 只有支持 top_p 的模型才添加该参数
  if (shouldUseTopP) {
    options.top_p = top_p
  }

  // [调试模式] - 开始
  console.log('\n/==================== OpenAI API 请求体 (调试模式) ====================/')
  console.log(`[调试] 请求时间: ${new Date().toISOString()}`)
  console.log(`[调试] 目标 URL: ${OPENAI_API_BASE_URL}`)
  // 为安全起见，只打印 API Key 的前5位和后4位
  console.log(`[调试] 使用 API Key: ${key.key.substring(0, 5)}...${key.key.slice(-4)}`)
  console.log('[调试] 完整请求负载 (Payload):')
  // 使用 console.dir 可以更好地格式化对象，并完整打印嵌套内容
  console.dir(options, { depth: null, colors: true })
  console.log('/=======================================================================/\n')
  // [调试模式] - 结束
  
  const apiResponse = await openai.chat.completions.create(options, {
    signal: abortSignal,
  })

  return apiResponse as AsyncIterable<OpenAI.ChatCompletionChunk>
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

  updateRoomChatModel(userId, options.room.roomId, model)

  const { message, uploadFileKeys, lastContext, process, systemMessage, temperature, top_p } = options
  let content: MessageContent = message
  let fileContext = ''
  //获取全局配置，判断当前模型是否在图片模型列表中
  const globalConfig = await getCacheConfig()
  const imageModelsStr = globalConfig.siteConfig.imageModels || ''
  //将配置字符串按逗号分割，去除空格，生成数组
  const imageModelList = imageModelsStr.split(/[,，]/).map(s => s.trim()).filter(Boolean)
  //判断当前模型是否包含在配置列表中 (支持模糊匹配，比如配置 gemini-3-pro 会匹配 gemini-3-pro-image)
  const isImage = imageModelList.some(m => model.includes(m))
  
  if (uploadFileKeys && uploadFileKeys.length > 0) {
    // 1. 先处理文本文件，读取内容拼接到 Prompt 中
    const textFiles = uploadFileKeys.filter(key => isTextFile(key))
    const imageFiles = uploadFileKeys.filter(key => !isTextFile(key)) // 假设非文本即图片，或者你可以加更严格的校验

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

    // 组合新的 Prompt
    const finalMessage = message + (fileContext ? `\n\nAttached Files Content:\n${fileContext}` : '')

    // 2. 如果有图片，使用多模态格式
    if (imageFiles.length > 0) {
      content = [
        {
          type: 'text',
          text: finalMessage, // 使用包含文件内容的文本
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
      // 3. 如果没有图片，只有文本文件，直接作为字符串发送
      content = finalMessage
    }
  }

  const abort = new AbortController()

  const customMessageId = generateMessageId()

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
      isImageModel: isImage,
    })
    processThreads.push({ userId, abort, messageId })

    let text = ''
    let chatIdRes = customMessageId
    let modelRes = ''
    let usageRes: OpenAI.Completions.CompletionUsage

    //分支处理
    if (isImage) {
      // --- 图片模型 (非流式处理) ---
      // 注意：这里 api 的类型不仅是 ChatCompletionChunk，也可能是 ChatCompletion
      const response = api as any // 简单处理类型推断问题
      const choice = response.choices[0]
      let rawContent = choice.message?.content || ''
      
      modelRes = response.model
      usageRes = response.usage

      // 自动包装 Markdown 图片语法
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
        // 伪造一个 detail 对象给前端
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
      // --- 文本模型 (流式处理) ---
      for await (const chunk of api as AsyncIterable<OpenAI.ChatCompletionChunk>) {
         // ... (原有的流式处理逻辑保持不变) ...
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
          message: {
            role: 'assistant',
            content: text,
          },
          finish_reason: 'stop',
          index: 0,
          logprobs: null,
        }],
        created: Date.now(),
        conversationId: lastContext.conversationId,
        model: modelRes,
        text,
        //最终返回的数据中包含自定义 ID
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
    // 保留了原有的复杂错误处理和重试机制
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
    // 保留了原有的 Abort 清理逻辑
    const index = processThreads.findIndex(d => d.userId === userId)
    if (index > -1)
      processThreads.splice(index, 1)
  }
}

// --- 以下所有辅助函数均保持不变 ---

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

    if (chatInfo.status !== Status.Normal) {
      return parentMessageId
        ? getMessageById(parentMessageId)
        : undefined
    }
    else {
      if (isPrompt) {
        let promptText = chatInfo.prompt
        const allFileKeys = chatInfo.images || [] // 数据库中我们将所有文件key都存在了 images 字段里
        
        const textFiles = allFileKeys.filter(k => isTextFile(k))
        const imageFiles = allFileKeys.filter(k => !isTextFile(k))

        // 1. 如果历史记录里有文本文件，重新读取内容并拼接到 Prompt 中
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

        // 2. 构建返回给 OpenAI 的 content
        let content: MessageContent = promptText

        // 如果还有图片文件，则转为多模态数组格式
        if (imageFiles.length > 0) {
          content = [
            {
              type: 'text',
              text: promptText,
            },
          ]
          for (const image of imageFiles) {
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
        return {
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
