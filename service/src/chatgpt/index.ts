// service/src/chatgpt/index.ts
import * as dotenv from 'dotenv'
import OpenAI from 'openai'
import jwt_decode from 'jwt-decode'
import { GoogleGenerativeAI, type Part } from '@google/generative-ai'
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
  }
  catch {
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
  }
  catch (e) {
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

// ===== DEBUG 开关 =====
const DEBUG_GEMINI_IMAGE = process.env.DEBUG_GEMINI_IMAGE === 'true'
function dlog(...args: any[]) {
  if (DEBUG_GEMINI_IMAGE) console.log(...args)
}
function summarizeUrl(u: string) {
  if (!u) return u
  if (u.startsWith('data:image/')) return `data:image(base64,len=${u.length})`
  return u
}
function summarizeUrls(urls: string[]) {
  return (urls || []).map(summarizeUrl)
}

// ===== 正则 =====
const DATA_URL_IMAGE_RE = /data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g
const HAS_DATA_URL_IMAGE_RE = /data:image\/[a-zA-Z0-9.+-]+;base64,/i

const MARKDOWN_IMAGE_RE =
  /!$$[^$$]*]$\s*<?([^)\s>]+)(?:\s+["'][^"']*["'][^)]*)?\s*>?\s*$/g

const HTML_IMAGE_RE = /<img[^>]*\ssrc=["']([^"']+)["'][^>]*>/gi

// ✅ 修复 “嵌套 markdown 图片”
// 把  ![image](![Generated Image](/uploads/xxx.png))  变成  ![Generated Image](/uploads/xxx.png)
const NESTED_UPLOAD_MD_RE = /!$$[^$$]*]$\s*!$$[^$$]*]\((\/uploads\/[^)]+)$\s*\)/g
function fixNestedUploadsMarkdownImages(text: string): string {
  if (!text) return text
  return text.replace(NESTED_UPLOAD_MD_RE, '![Generated Image]($1)')
}

function extractImageUrlsFromText(text: string): { cleanedText: string; urls: string[] } {
  if (!text) return { cleanedText: '', urls: [] }

  // 先修复嵌套，避免抽取失败
  const fixed = fixNestedUploadsMarkdownImages(text)

  const urls: string[] = []
  let cleaned = fixed

  cleaned = cleaned.replace(MARKDOWN_IMAGE_RE, (_m, url) => {
    if (url) urls.push(url)
    return '[Image]'
  })

  cleaned = cleaned.replace(HTML_IMAGE_RE, (_m, url) => {
    if (url) urls.push(url)
    return '[Image]'
  })

  DATA_URL_IMAGE_RE.lastIndex = 0
  const rawDataUrls = cleaned.match(DATA_URL_IMAGE_RE)
  if (rawDataUrls?.length) urls.push(...rawDataUrls)

  DATA_URL_IMAGE_RE.lastIndex = 0
  cleaned = cleaned.replace(DATA_URL_IMAGE_RE, '[Image]')

  return { cleanedText: cleaned, urls }
}

async function extractImageUrlsFromMessageContent(content: MessageContent): Promise<{ cleanedText: string; urls: string[] }> {
  const urls: string[] = []
  let cleanedText = ''

  if (typeof content === 'string') {
    const r = extractImageUrlsFromText(content)
    cleanedText = r.cleanedText
    urls.push(...r.urls)
    if (!cleanedText?.trim() && urls.length) cleanedText = '[Image]'
    return { cleanedText, urls }
  }

  if (Array.isArray(content)) {
    for (const p of content as any[]) {
      if (p?.type === 'text' && p.text) {
        const r = extractImageUrlsFromText(p.text)
        if (r.cleanedText) cleanedText += (cleanedText ? '\n' : '') + r.cleanedText
        if (r.urls.length) urls.push(...r.urls)
      }
      else if (p?.type === 'image_url' && p.image_url?.url) {
        urls.push(p.image_url.url)
      }
    }
    if (!cleanedText?.trim() && urls.length) cleanedText = '[Image]'
    return { cleanedText, urls }
  }

  return { cleanedText: '', urls: [] }
}

function dedupeUrlsPreserveOrder(urls: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const u of urls) {
    if (!u) continue
    if (seen.has(u)) continue
    seen.add(u)
    out.push(u)
  }
  return out
}

async function imageUrlToGeminiInlinePart(urlStr: string): Promise<Part | null> {
  if (urlStr.startsWith('data:')) {
    const [header, base64Data] = urlStr.split(',')
    if (!header || !base64Data) return null
    const mimeMatch = header.match(/:(.*?);/)
    const mimeType = mimeMatch ? mimeMatch[1] : 'image/png'
    return { inlineData: { mimeType, data: base64Data } }
  }

  if (urlStr.includes('/uploads/')) {
    const filename = path.basename(urlStr)
    const fileData = await getFileBase64(filename)
    if (!fileData) {
      dlog('[GeminiImage][DEBUG] getFileBase64 failed for uploads url:', urlStr, '-> filename:', filename)
      return null
    }
    return { inlineData: { mimeType: fileData.mime, data: fileData.data } }
  }

  dlog('[GeminiImage][DEBUG] imageUrlToGeminiInlinePart: unsupported url:', summarizeUrl(urlStr))
  return null
}

// ===== dataURL 落盘 =====
const MARKDOWN_DATA_URL_IMAGE_RE =
  /!$$([^$$]*)\]$\s*(data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+)\s*$/g

async function saveDataUrlToUploads(dataUrl: string): Promise<string | null> {
  try {
    const [header, base64Data] = dataUrl.split(',')
    if (!header || !base64Data) return null
    const mimeMatch = header.match(/data:([^;]+);base64/i)
    const mime = mimeMatch?.[1] || 'image/png'
    const buffer = Buffer.from(base64Data, 'base64')
    const ext = mime.split('/')[1] || 'png'
    const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
    await ensureUploadDir()
    await fs.writeFile(path.join(UPLOAD_DIR, filename), buffer)
    return `/uploads/${filename}`
  }
  catch (e) {
    console.error('[GeminiImage] saveDataUrlToUploads failed:', e)
    return null
  }
}

async function rewriteDataUrlImagesToUploads(text: string): Promise<string> {
  if (!text) return text

  let out = text
  let convertedCount = 0

  // ✅ 先修复嵌套（防止越写越乱）
  out = fixNestedUploadsMarkdownImages(out)

  // 1) markdown 包裹 dataURL：![alt](data:...) => ![alt](/uploads/xxx)
  {
    const re = new RegExp(MARKDOWN_DATA_URL_IMAGE_RE)
    let result = ''
    let lastIndex = 0
    let m: RegExpExecArray | null
    while ((m = re.exec(out)) !== null) {
      result += out.slice(lastIndex, m.index)
      const alt = m[1] || 'image'
      const dataUrl = m[2]
      const uploadsUrl = await saveDataUrlToUploads(dataUrl)
      if (uploadsUrl) {
        result += `![${alt}](${uploadsUrl})`
        convertedCount++
      }
      else {
        result += m[0]
      }
      lastIndex = re.lastIndex
    }
    result += out.slice(lastIndex)
    out = result
  }

  // 2) 裸 dataURL：直接替换为 [Image]（避免插入新的 markdown 再被包一层造成嵌套）
  {
    DATA_URL_IMAGE_RE.lastIndex = 0
    if (DATA_URL_IMAGE_RE.test(out)) {
      DATA_URL_IMAGE_RE.lastIndex = 0
      out = out.replace(DATA_URL_IMAGE_RE, '[Image]')
    }
    DATA_URL_IMAGE_RE.lastIndex = 0
  }

  // 再修复一次嵌套（兜底）
  out = fixNestedUploadsMarkdownImages(out)

  if (DEBUG_GEMINI_IMAGE) {
    console.log('[GeminiImage][DEBUG] rewriteDataUrlImagesToUploads convertedCount =', convertedCount)
  }

  return out
}

async function buildGeminiHistoryFromLastMessageId(params: {
  lastMessageId?: string
  maxContextCount: number
  forGeminiImageModel: boolean
}): Promise<{ history: any[]; contextImageUrls: string[] }> {
  const { lastMessageId: startId, maxContextCount } = params

  const messages: { role: 'user' | 'model'; parts: Part[] }[] = []
  const collectedUrls: string[] = []

  let lastMessageId = startId
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId) break
    const msg = await getMessageById(lastMessageId)
    if (!msg) break

    const role = (msg.role === 'assistant' ? 'model' : 'user') as 'user' | 'model'

    const { cleanedText, urls } = await extractImageUrlsFromMessageContent(msg.text)
    if (urls.length) collectedUrls.push(...urls)

    if (DEBUG_GEMINI_IMAGE) {
      const preview = typeof msg.text === 'string'
        ? msg.text.slice(0, 240)
        : JSON.stringify(msg.text).slice(0, 240)
      dlog('[GeminiImage][DEBUG] history.msg', {
        i,
        role,
        extractedUrls: summarizeUrls(urls),
        preview,
        cleanedTextPreview: cleanedText.slice(0, 160),
      })
    }

    const safeText = (cleanedText && cleanedText.trim())
      ? cleanedText
      : (urls.length ? '[Image]' : '[Empty]')

    messages.push({ role, parts: [{ text: safeText }] })
    lastMessageId = msg.parentMessageId
  }

  messages.reverse()
  collectedUrls.reverse()

  const deduped = dedupeUrlsPreserveOrder(collectedUrls)

  dlog('[GeminiImage][DEBUG] buildGeminiHistoryFromLastMessageId collectedUrls =', summarizeUrls(collectedUrls))
  dlog('[GeminiImage][DEBUG] buildGeminiHistoryFromLastMessageId deduped =', summarizeUrls(deduped))
  dlog('[GeminiImage][DEBUG] buildGeminiHistoryFromLastMessageId history.len =', messages.length)

  return {
    history: messages.map(m => ({ role: m.role, parts: m.parts })),
    contextImageUrls: deduped,
  }
}

// ===== 下面基本保持你原项目结构（OpenAI initApi 等）=====
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
    timeout: config.timeoutMs,
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
      DATA_URL_IMAGE_RE.lastIndex = 0
      safeContent = safeContent.replace(DATA_URL_IMAGE_RE, '[Image Data Removed]')
    }

    messages.push({
      role: message.role as any,
      content: safeContent,
    })
    lastMessageId = message.parentMessageId
  }
  if (systemMessage) {
    messages.push({ role: 'system', content: systemMessage })
  }
  messages.reverse()
  messages.push({ role: 'user', content })

  const enableStream = !isImageModel

  const options: OpenAI.ChatCompletionCreateParams = {
    model,
    stream: enableStream,
    stream_options: enableStream ? { include_usage: true } : undefined,
    messages,
  }
  options.temperature = finalTemperature
  if (shouldUseTopP) options.top_p = top_p

  console.log(`[OpenAI] Request Model: ${model}, URL: ${OPENAI_API_BASE_URL}`)

  const apiResponse = await openai.chat.completions.create(options, { signal: abortSignal })
  return apiResponse as AsyncIterable<OpenAI.ChatCompletionChunk>
}

const processThreads: { userId: string; abort: AbortController; messageId: string; roomId: number }[] = []

async function chatReplyProcess(options: RequestOptions): Promise<{ message: string; data: ChatResponse; status: string }> {
  const userId = options.user._id.toString()
  const messageId = options.messageId
  const roomId = options.room.roomId
  const abort = new AbortController()
  const customMessageId = generateMessageId()

  const existingIdx = processThreads.findIndex(t => t.userId === userId)
  if (existingIdx > -1) processThreads.splice(existingIdx, 1)

  console.log(`[DEBUG] Pushing thread for userId: ${userId}, roomId: ${roomId}`)
  processThreads.push({ userId, abort, messageId, roomId })

  await ensureUploadDir()
  const model = options.room.chatModel
  const key = await getRandomApiKey(options.user, model, options.room.accountId)
  const maxContextCount = options.user.advanced.maxContextCount ?? 20
  if (!key) throw new Error('没有对应的 apikeys 配置，请再试一次。')

  updateRoomChatModel(userId, options.room.roomId, model)

  const { message, uploadFileKeys, lastContext, process, systemMessage, temperature, top_p } = options
  let content: MessageContent = message
  let fileContext = ''
  const globalConfig = await getCacheConfig()
  const imageModelsStr = globalConfig.siteConfig.imageModels || ''
  const imageModelList = imageModelsStr.split(/[,，]/).map(s => s.trim()).filter(Boolean)

  const isImage = imageModelList.some(m => model.includes(m))
  const isGeminiImageModel = isImage && model.includes('gemini')

  if (DEBUG_GEMINI_IMAGE) {
    dlog('[GeminiImage][DEBUG] incoming uploadFileKeys =', uploadFileKeys)
    dlog('[GeminiImage][DEBUG] incoming message.len =', message?.length ?? 0)
  }

  if (uploadFileKeys?.length) {
    const textFiles = uploadFileKeys.filter(key => isTextFile(key))
    const imageFiles = uploadFileKeys.filter(key => isImageFile(key))

    if (textFiles.length) {
      for (const fileKey of textFiles) {
        try {
          const filePath = path.join(UPLOAD_DIR, stripTypePrefix(fileKey))
          await fs.access(filePath)
          const fileContent = await fs.readFile(filePath, 'utf-8')
          fileContext += `\n\n--- File Start: ${stripTypePrefix(fileKey)} ---\n${fileContent}\n--- File End ---\n`
        }
        catch (e) {
          console.error(`Error reading text file ${fileKey}`, e)
          fileContext += `\n\n[System Error: File ${fileKey} not found or unreadable]\n`
        }
      }
    }

    const finalMessage = message + (fileContext ? `\n\nAttached Files Content:\n${fileContext}` : '')

    if (imageFiles.length) {
      content = [{ type: 'text', text: finalMessage }]
      for (const uploadFileKey of imageFiles) {
        content.push({
          type: 'image_url',
          image_url: { url: await convertImageUrl(stripTypePrefix(uploadFileKey)) },
        })
      }
    }
    else {
      content = finalMessage
    }
  }

  try {
    if (isGeminiImageModel) {
      const baseUrl = isNotEmptyString(key.baseUrl) ? key.baseUrl : undefined

      const { history, contextImageUrls } = await buildGeminiHistoryFromLastMessageId({
        lastMessageId: lastContext.parentMessageId,
        maxContextCount,
        forGeminiImageModel: true,
      })

      const { cleanedText: currentCleanedText, urls: currentUrls } = await extractImageUrlsFromMessageContent(content)

      const allImageUrls = dedupeUrlsPreserveOrder([...contextImageUrls, ...currentUrls])

      dlog('[GeminiImage][DEBUG] model =', model)
      dlog('[GeminiImage][DEBUG] baseUrl =', baseUrl)
      dlog('[GeminiImage][DEBUG] history.len =', history.length)
      dlog('[GeminiImage][DEBUG] contextImageUrls =', summarizeUrls(contextImageUrls))
      dlog('[GeminiImage][DEBUG] currentUrls =', summarizeUrls(currentUrls))
      dlog('[GeminiImage][DEBUG] allImageUrls =', summarizeUrls(allImageUrls))
      dlog('[GeminiImage][DEBUG] currentCleanedText.len =', currentCleanedText?.length ?? 0)

      const inputParts: Part[] = []
      for (const u of allImageUrls) {
        const imgPart = await imageUrlToGeminiInlinePart(u)
        if (imgPart) inputParts.push(imgPart)
      }

      const finalPromptText = (currentCleanedText && currentCleanedText.trim())
        ? currentCleanedText.trim()
        : (inputParts.length ? '请基于以上所有图片继续生成/修改。' : '请根据要求生成图片。')

      inputParts.push({ text: finalPromptText })

      const genAI = new GoogleGenerativeAI(key.key)
      const requestOptions: any = {}
      if (globalConfig.timeoutMs) requestOptions.timeout = globalConfig.timeoutMs
      if (baseUrl) requestOptions.baseUrl = baseUrl.replace(/\/+$/, '')

      const geminiModel = genAI.getGenerativeModel({
        model,
        systemInstruction: systemMessage ? { parts: [{ text: systemMessage }], role: 'system' } : undefined,
      }, requestOptions)

      const chatSession = geminiModel.startChat({ history })
      const result = await chatSession.sendMessage(inputParts)
      const response = await result.response

      if (!response.candidates || response.candidates.length === 0) {
        throw new Error(`[Gemini] Empty candidates. usage=${JSON.stringify(response.usageMetadata ?? {})}`)
      }

      let text = ''
      const parts = response.candidates?.[0]?.content?.parts ?? []

      for (const part of parts as any[]) {
        if (part.text) text += part.text
      }

      if (!text) text = '[Gemini] Success but no text returned.'

      // 先落盘 dataURL
      text = await rewriteDataUrlImagesToUploads(text)
      // 再修复嵌套（极重要）
      text = fixNestedUploadsMarkdownImages(text)

      const usageRes: any = response.usageMetadata
        ? {
            prompt_tokens: response.usageMetadata.promptTokenCount ?? 0,
            completion_tokens: response.usageMetadata.candidatesTokenCount ?? 0,
            total_tokens: response.usageMetadata.totalTokenCount ?? 0,
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
      }
      else {
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
          usage: response.usage,
        } as any,
      })
    }
    else {
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
        choices: [{ message: { role: 'assistant', content: text }, finish_reason: 'stop', index: 0, logprobs: null }],
        created: Date.now(),
        conversationId: lastContext.conversationId,
        model: modelRes,
        text,
        id: chatIdRes,
        detail: { usage: usageRes && { ...usageRes, estimated: false } },
      },
    })
  }
  catch (error: any) {
    const code = error.statusCode
    globalThis.console.error(error)
    if (Reflect.has(ErrorCodeMessage, code))
      return sendResponse({ type: 'Fail', message: ErrorCodeMessage[code] })
    return sendResponse({ type: 'Fail', message: error.message ?? 'Please check the back-end console' })
  }
  finally {
    const index = processThreads.findIndex(d => d.userId === userId)
    if (index > -1) processThreads.splice(index, 1)
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

  if (!chatInfo) return undefined

  const parentMessageId = isPrompt
    ? chatInfo.options.parentMessageId
    : `prompt_${id}`

  if (chatInfo.status !== Status.Normal) {
    return parentMessageId ? getMessageById(parentMessageId) : undefined
  }

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
        }
        catch { }
      }
      promptText += (fileContext ? `\n\n[Attached Files History]:\n${fileContext}` : '')
    }

    if (promptText && typeof promptText === 'string') {
      DATA_URL_IMAGE_RE.lastIndex = 0
      promptText = promptText.replace(DATA_URL_IMAGE_RE, '[Image Data Removed]')
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

  // assistant
  let responseText = chatInfo.response || ''
  if (responseText && typeof responseText === 'string') {
    // 先修复嵌套
    responseText = fixNestedUploadsMarkdownImages(responseText)

    // 如果还带 dataURL，先落盘
    if (HAS_DATA_URL_IMAGE_RE.test(responseText)) {
      responseText = await rewriteDataUrlImagesToUploads(responseText)
      responseText = fixNestedUploadsMarkdownImages(responseText)
    }

    // 再替换残留 dataURL
    DATA_URL_IMAGE_RE.lastIndex = 0
    responseText = responseText.replace(DATA_URL_IMAGE_RE, '[Image History]')
  }

  return {
    id,
    conversationId: chatInfo.options.conversationId,
    parentMessageId,
    role: 'assistant',
    text: responseText,
  }
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
  return unsedKeys[Math.floor(Math.random() * unsedKeys.length)]
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
  catch {
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
