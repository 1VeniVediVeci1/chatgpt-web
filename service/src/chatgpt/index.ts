import * as dotenv from 'dotenv'
import OpenAI from 'openai'
import jwt_decode from 'jwt-decode'
import { GoogleGenerativeAI, type ChatSession, type Part } from '@google/generative-ai'
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
import * as fsData from 'fs' 

const MODEL_CONFIGS: Record<string, { supportTopP: boolean; defaultTemperature?: number }> = {
  'gpt-5-search-api': { supportTopP: false, defaultTemperature: 0.8 },
}

const UPLOAD_DIR = path.resolve(process.cwd(), 'uploads')

async function ensureUploadDir() {
  try {
    await fs.access(UPLOAD_DIR)
  } catch {
    await fs.mkdir(UPLOAD_DIR, { recursive: true })
  }
}

function stripTypePrefix(key: string): string {
  return key.replace(/^(img:|txt:)/, '')
}

function isTextFile(filename: string): boolean {
  if (filename.startsWith('txt:')) return true
  if (filename.startsWith('img:')) return false
  const realName = stripTypePrefix(filename)
  const ext = path.extname(realName).toLowerCase()
  const textExtensions = ['.txt', '.md', '.json', '.csv', '.js', '.ts', '.py', '.java', '.html', '.css', '.xml', '.yml', '.yaml', '.log', '.ini', '.config']
  return textExtensions.includes(ext)
}

function isImageFile(filename: string): boolean {
  if (filename.startsWith('img:')) return true
  if (filename.startsWith('txt:')) return false
  const realName = stripTypePrefix(filename)
  const ext = path.extname(realName).toLowerCase()
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.heic', '.bmp']
  return imageExtensions.includes(ext)
}

async function getFileBase64(filename: string): Promise<{ mime: string; data: string } | null> {
  try {
    const realFilename = stripTypePrefix(filename)
    const filePath = path.join(UPLOAD_DIR, realFilename)
    await fs.access(filePath) 
    const buffer = await fs.readFile(filePath)
    const ext = path.extname(realFilename).toLowerCase().replace('.', '')
    let mime = 'image/png'
    if (ext === 'jpg' || ext === 'jpeg') mime = 'image/jpeg'
    else if (ext === 'webp') mime = 'image/webp'
    else if (ext === 'gif') mime = 'image/gif'
    else if (ext === 'heic') mime = 'image/heic'
    return { mime, data: buffer.toString('base64') }
  } catch (e) {
    console.error(`[File Read Error] ${filename}:`, e)
    return null
  }
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
const geminiChats = new Map<string, ChatSession>()

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
    if (!lastMessageId) break
    const message = await getMessageById(lastMessageId)
    if (!message) break
    
    let safeContent = message.text
    if (typeof safeContent === 'string') {
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

  console.log(`[OpenAI] Request Model: ${model}, URL: ${OPENAI_API_BASE_URL}`)

  const apiResponse = await openai.chat.completions.create(options, {
    signal: abortSignal,
  })

  return apiResponse as AsyncIterable<OpenAI.ChatCompletionChunk>
}

const processThreads: { userId: string; abort: AbortController; messageId: string; roomId: number }[] = []

function getGeminiChatSession(
  apiKey: string,
  modelName: string,
  baseUrl: string | undefined,
  sessionKey: string,
  systemInstruction?: string
): ChatSession {
  let chatSession = geminiChats.get(sessionKey)

  if (!chatSession) {
    console.log(`[Gemini SDK] Creating new session for ${sessionKey}`)
    const genAI = new GoogleGenerativeAI(apiKey)
    
    const requestOptions: any = {}
    if (baseUrl) {
      requestOptions.baseUrl = baseUrl.replace(/\/+$/, '')
    }

    const model = genAI.getGenerativeModel({ 
      model: modelName,
      systemInstruction: systemInstruction ? { parts: [{ text: systemInstruction }], role: "system" } : undefined
    }, requestOptions)

    chatSession = model.startChat({
      history: [], 
    })
    geminiChats.set(sessionKey, chatSession)
  }

  return chatSession
}

async function chatReplyProcess(options: RequestOptions): Promise<{ message: string; data: ChatResponse; status: string }> {
  await ensureUploadDir()
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
  const isGeminiImageModel = isImage && model.includes('gemini')

  if (uploadFileKeys && uploadFileKeys.length > 0) {
    const textFiles = uploadFileKeys.filter(key => isTextFile(key))
    const imageFiles = uploadFileKeys.filter(key => isImageFile(key))

    if (textFiles.length > 0) {
      for (const fileKey of textFiles) {
        try {
          const filePath = path.join(UPLOAD_DIR, stripTypePrefix(fileKey))
          await fs.access(filePath)
          const fileContent = await fs.readFile(filePath, 'utf-8')
          fileContext += `\n\n--- File Start: ${stripTypePrefix(fileKey)} ---\n${fileContent}\n--- File End ---\n`
        } catch (e) {
          console.error(`Error reading text file ${fileKey}`, e)
          fileContext += `\n\n[System Error: File ${fileKey} not found or unreadable]\n`
        }
      }
    }

    const finalMessage = message + (fileContext ? `\n\nAttached Files Content:\n${fileContext}` : '')

    if (imageFiles.length > 0) {
      content = [
        { type: 'text', text: finalMessage },
      ]
      for (const uploadFileKey of imageFiles) {
        content.push({
          type: 'image_url',
          image_url: { url: await convertImageUrl(stripTypePrefix(uploadFileKey)) },
        })
      }
    } else {
      content = finalMessage
    }
  }

  const abort = new AbortController()
  const customMessageId = generateMessageId()

  try {
    // ====== ① Gemini 图片模型 ======
    if (isGeminiImageModel) {
      const baseUrl = isNotEmptyString(key.baseUrl) ? key.baseUrl : undefined
      const sessionKey = `${userId}:${options.room.roomId}:${model}`
      
      const chatSession = getGeminiChatSession(key.key, model, baseUrl, sessionKey, systemMessage)

      const inputParts: Part[] = []
      
      let promptText = ''
      if (typeof content === 'string') {
        promptText = content
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if ((part as any).type === 'text' && (part as any).text) {
            promptText += `${(part as any).text}\n`
          }
        }
      }
      if (promptText) inputParts.push({ text: promptText })

      if (Array.isArray(content)) {
        for (const part of content) {
          if ((part as any).type === 'image_url' && (part as any).image_url?.url) {
            const urlStr = (part as any).image_url.url as string
            if (urlStr.startsWith('data:')) {
                const [header, base64Data] = urlStr.split(',')
                if (header && base64Data) {
                  const mimeMatch = header.match(/:(.*?);/)
                  const mimeType = mimeMatch ? mimeMatch[1] : 'image/png'
                  inputParts.push({ inlineData: { mimeType, data: base64Data } })
                }
            } else if (urlStr.includes('/uploads/')) {
                const filename = path.basename(urlStr)
                const fileData = await getFileBase64(filename)
                if (fileData) {
                    inputParts.push({ inlineData: { mimeType: fileData.mime, data: fileData.data } })
                }
            }
          }
        }
      }

      processThreads.push({ userId, abort, messageId, roomId: options.room.roomId })

      const result = await chatSession.sendMessage(inputParts)
      const response = await result.response
      
      let text = ''
      const parts = response.candidates?.[0]?.content?.parts ?? []

      for (const part of parts) {
        if (part.text) text += part.text
        if (part.inlineData?.data) {
          const mime = part.inlineData.mimeType || 'image/png'
          const base64 = part.inlineData.data as string
          const buffer = Buffer.from(base64, 'base64')
          const ext = mime.split('/')[1] || 'png'
          const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
          const filePath = path.join(UPLOAD_DIR, filename)
          await fs.writeFile(filePath, buffer)
          const prefix = text ? '\n\n' : ''
          text += `${prefix}![Generated Image](/uploads/${filename})`
        }
      }

      if (!text) text = JSON.stringify(response)

      const usageRes: any = response.usageMetadata
        ? {
            prompt_tokens: response.usageMetadata.promptTokenCount ?? 0,
            completion_tokens: response.usageMetadata.candidatesTokenCount ?? 0,
            total_tokens: response.usageMetadata.totalTokenCount ?? 0, estimated: true,
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
          choices: [{ message: { role: 'assistant', content: text }, finish_reason: 'stop', index: 0, logprobs: null }],
          created: Date.now(),
          conversationId: lastContext.conversationId,
          model, text, id: customMessageId,
          detail: { usage: usageRes },
        },
      })
    }

    // ====== ② 其它模型 ======
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
    
    processThreads.push({ userId, abort, messageId, roomId: options.room.roomId })

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
        detail: { choices: [{ finish_reason: 'stop', index: 0, logprobs: null, message: choice.message }], created: response.created, id: response.id, model: response.model, object: 'chat.completion', usage: response.usage } as any 
      })

    } else {
      // ★★★ 这里加了循环计数日志 ★★★
      let loopCount = 0; 
      for await (const chunk of api as AsyncIterable<OpenAI.ChatCompletionChunk>) {
         loopCount++;
         // 每 5 个 chunk 打印一次，防止刷屏太快
         if (loopCount % 5 === 0) console.log(`[DEBUG] 正在生成第 ${loopCount} 个片段... [${userId}]`);

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
        choices: [{ message: { role: 'assistant', content: text }, finish_reason: 'stop', index: 0, logprobs: null }],
        created: Date.now(),
        conversationId: lastContext.conversationId,
        model: modelRes, text, id: chatIdRes,
        detail: { usage: usageRes && { ...usageRes, estimated: false } },
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
    
    if (isGeminiImageModel) {
        const sessionKey = `${userId}:${options.room.roomId}:${model}`
        geminiChats.delete(sessionKey)
    }

    if (Reflect.has(ErrorCodeMessage, code))
      return sendResponse({ type: 'Fail', message: ErrorCodeMessage[code] })
    return sendResponse({ type: 'Fail', message: error.message ?? 'Please check the back-end console' })
  }
  finally {
    console.log(`[DEBUG] 任务彻底结束，清理线程状态。UserId: ${userId}`)
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
        const imageFiles = allFileKeys.filter(k => isImageFile(k)) 

        if (textFiles.length > 0) {
          let fileContext = ''
          for (const fileKey of textFiles) {
            try {
              const filePath = path.join(UPLOAD_DIR, stripTypePrefix(fileKey))
              const fileContent = await fs.readFile(filePath, 'utf-8')
              fileContext += `\n\n--- Context File: ${stripTypePrefix(fileKey)} ---\n${fileContent}\n--- End File ---\n`
            } catch (e) {}
          }
          promptText += (fileContext ? `\n\n[Attached Files History]:\n${fileContext}` : '')
        }
        
        if (promptText && typeof promptText === 'string') {
           promptText = promptText.replace(/!\[.*?\]\(data:image\/.*?;base64,.*?\)/g, '[Image Data Removed]')
        }

        let content: MessageContent = promptText

        if (imageFiles.length > 0) {
          content = [{ type: 'text', text: promptText }]
          for (const image of imageFiles) {
            content.push({
              type: 'image_url',
              image_url: { url: await convertImageUrl(stripTypePrefix(image)) },
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
        let responseText = chatInfo.response || ''
        if (responseText.includes('data:image') && responseText.includes('base64')) {
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

export function getChatProcessState(userId: string) {
  const thread = processThreads.find(d => d.userId === userId)
  if (thread) {
    return {
      isProcessing: true,
      roomId: thread.roomId,
      messageId: thread.messageId,
    }
  }
  return { isProcessing: false, roomId: null, messageId: null }
}

export { chatReplyProcess, chatConfig, containsSensitiveWords }
