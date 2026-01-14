import * as dotenv from 'dotenv'
import OpenAI from 'openai'
import jwt_decode from 'jwt-decode'
import { GoogleGenAI } from '@google/genai'
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

/**
 * 兼容 Gemini 构造的 parts
 */
type GeminiPart = {
  text?: string
  inlineData?: {
    mimeType: string
    data: string
  }
}

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

// ===================== Helper: Extract Images =====================
const DATA_URL_IMAGE_CAPTURE_RE = /data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)/g
const MARKDOWN_IMAGE_RE = /!$$[^$$]*]$\s*([\s\S]*?)\s*$/g
const UPLOADS_URL_RE = /(\/uploads\/[^)\s>"']+\.(?:png|jpe?g|webp|gif|bmp|heic))/gi
const HTML_IMAGE_RE = /<img[^>]*\ssrc=["']([^"']+)["'][^>]*>/gi
const DATA_URL_IMAGE_RE = /data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g

async function replaceDataUrlImagesWithUploads(text: string): Promise<{
  text: string
  saved: Array<{ mime: string; filename: string; bytes: number }>
}> {
  if (!text) return { text: '', saved: [] }

  const saved: Array<{ mime: string; filename: string; bytes: number }> = []
  let out = ''
  let lastIndex = 0

  const matches = text.matchAll(DATA_URL_IMAGE_CAPTURE_RE)
  for (const m of matches) {
    const full = m[0]
    const mime = m[1]
    const base64 = m[2]
    const idx = m.index ?? -1
    if (idx < 0) continue

    out += text.slice(lastIndex, idx)

    const buffer = Buffer.from(base64, 'base64')
    const ext = mime.split('/')[1] || 'png'
    const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
    const filePath = path.join(UPLOAD_DIR, filename)
    await fs.writeFile(filePath, buffer)

    saved.push({ mime, filename, bytes: buffer.length })
    out += `/uploads/${filename}`

    lastIndex = idx + full.length
  }

  out += text.slice(lastIndex)
  return { text: out, saved }
}

function extractImageUrlsFromText(text: string): { cleanedText: string; urls: string[] } {
  if (!text) return { cleanedText: '', urls: [] }

  const urls: string[] = []
  let cleaned = text

  cleaned = cleaned.replace(MARKDOWN_IMAGE_RE, (_m, inside) => {
    const raw = String(inside ?? '').trim()
    if (raw) {
      const firstToken = raw.split(/\s+/)[0] || ''
      const url = firstToken.replace(/^<|>$/g, '').trim()
      if (url) urls.push(url)
    }
    return '[Image]'
  })

  cleaned = cleaned.replace(HTML_IMAGE_RE, (_m, url) => {
    if (url) urls.push(url)
    return '[Image]'
  })

  const rawDataUrls = cleaned.match(DATA_URL_IMAGE_RE)
  if (rawDataUrls?.length) urls.push(...rawDataUrls)
  cleaned = cleaned.replace(DATA_URL_IMAGE_RE, '[Image]')

  const uploadUrls = cleaned.match(UPLOADS_URL_RE)
  if (uploadUrls?.length) urls.push(...uploadUrls)
  cleaned = cleaned.replace(UPLOADS_URL_RE, '[Image]')

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

async function imageUrlToGeminiInlinePart(urlStr: string): Promise<GeminiPart | null> {
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
    if (!fileData) return null
    return { inlineData: { mimeType: fileData.mime, data: fileData.data } }
  }

  return null
}

// ===== Image Binding Type =====
type ImageBinding = {
  tag: 'Image#1' | 'Image#2'
  kind: 'initial' | 'latest'
  url: string
}

type HistoryMessageMeta = {
  role: 'user' | 'model'
  cleanedText: string
  urls: string[]
  messageId: string
}

function applyImageBindingsToCleanedText(cleanedText: string, urls: string[], bindings: ImageBinding[]): string {
  const map = new Map<string, ImageBinding['tag']>()
  for (const b of bindings) map.set(b.url, b.tag)

  let out = cleanedText || ''
  for (const u of urls) {
    const tag = map.get(u)
    const token = tag ? `[${tag}]` : '[Image]'
    out = out.replace('[Image]', token)
  }

  if (!out.trim() && urls.length) {
    out = urls.map(u => (map.get(u) ? `[${map.get(u)}]` : '[Image]')).join(' ')
  }

  return out
}

async function buildGeminiHistoryFromLastMessageId(params: {
  lastMessageId?: string
  maxContextCount: number
  forGeminiImageModel: boolean
}): Promise<{ metas: HistoryMessageMeta[]; allUrls: string[] }> {
  const { lastMessageId: startId, maxContextCount, forGeminiImageModel } = params

  const metasRev: HistoryMessageMeta[] = []
  const urlsRev: string[] = []

  let lastMessageId = startId
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId) break
    const msg = await getMessageById(lastMessageId)
    if (!msg) break

    const role = (msg.role === 'assistant' ? 'model' : 'user') as 'user' | 'model'

    if (forGeminiImageModel) {
      const { cleanedText, urls } = await extractImageUrlsFromMessageContent(msg.text)
      if (urls.length) urlsRev.push(...urls)

      metasRev.push({
        role,
        cleanedText: cleanedText?.trim() ? cleanedText : (urls.length ? '[Image]' : '[Empty]'),
        urls,
        messageId: msg.id,
      })
    }
    else {
      metasRev.push({
        role,
        cleanedText: typeof msg.text === 'string' ? msg.text : '[Complex Content]',
        urls: [],
        messageId: msg.id,
      })
    }

    lastMessageId = msg.parentMessageId
  }

  const metas = metasRev.reverse()
  const allUrls = dedupeUrlsPreserveOrder(urlsRev.reverse())
  return { metas, allUrls }
}

function selectTwoImages(metas: HistoryMessageMeta[], allUrls: string[], currentUrls: string[]): ImageBinding[] {
  const combinedAll = dedupeUrlsPreserveOrder([...allUrls, ...currentUrls])
  if (!combinedAll.length) return []

  const allInOrder = combinedAll

  const earliestUserUrl = (() => {
    for (const m of metas) {
      if (m.role !== 'user') continue
      for (const u of m.urls) return u
    }
    return currentUrls[0]
  })()

  const initial = earliestUserUrl || allInOrder[0]

  const latestModelUploads = (() => {
    for (let i = metas.length - 1; i >= 0; i--) {
      const m = metas[i]
      if (m.role !== 'model') continue
      for (let j = m.urls.length - 1; j >= 0; j--) {
        const u = m.urls[j]
        if (u && u.includes('/uploads/')) return u
      }
    }
    return undefined
  })()

  const latest = latestModelUploads || allInOrder[allInOrder.length - 1]

  const bindings: ImageBinding[] = []
  if (initial) bindings.push({ tag: 'Image#1', kind: 'initial', url: initial })
  if (latest && latest !== initial) bindings.push({ tag: 'Image#2', kind: 'latest', url: latest })
  return bindings
}

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
}: any) {
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

    let safeContent = message.text as string
    if (typeof safeContent === 'string') {
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

  try {
    const siteCfg = config.siteConfig
    const reasoningModelsStr = siteCfg?.reasoningModels || ''
    const reasoningEffort = siteCfg?.reasoningEffort || 'medium'

    const reasoningModelList = reasoningModelsStr
      .split(/[,，]/)
      .map(s => s.trim())
      .filter(Boolean)

    const isReasoningModel = reasoningModelList.includes(model)

    if (isReasoningModel && reasoningEffort && reasoningEffort !== 'none') {
      ; (options as any).reasoning_effort = reasoningEffort
    }
  }
  catch (e) {
    console.error('[OpenAI] set reasoning_effort failed:', e)
  }

  const apiResponse = await openai.chat.completions.create(options, {
    signal: abortSignal,
  })

  return apiResponse
}

// ✅ 修改：ProcessThreads 支持 Key (userId:roomId)
type ProcessThread = { key: string; userId: string; roomId: number; abort: AbortController; messageId: string }
const processThreads: ProcessThread[] = []
const makeThreadKey = (userId: string, roomId: number) => `${userId}:${roomId}`

async function chatReplyProcess(options: RequestOptions): Promise<{ message: string; data: ChatResponse; status: string }> {
  const userId = options.user._id.toString()
  const messageId = options.messageId
  const roomId = options.room.roomId
  const abort = new AbortController()
  const customMessageId = generateMessageId()

  const threadKey = makeThreadKey(userId, roomId)

  // ✅ 只清除当前 Room 的旧任务，不清除其他 Room 的
  const existingIdx = processThreads.findIndex(t => t.key === threadKey)
  if (existingIdx > -1) {
    processThreads[existingIdx].abort.abort()
    processThreads.splice(existingIdx, 1)
  }

  processThreads.push({ key: threadKey, userId, abort, messageId, roomId })

  await ensureUploadDir()
  const model = options.room.chatModel
  const key = await getRandomApiKey(options.user, model, options.room.accountId)
  const maxContextCount = options.user.advanced.maxContextCount ?? 20
  if (!key) {
    const idx = processThreads.findIndex(d => d.key === threadKey)
    if (idx > -1) processThreads.splice(idx, 1)
    throw new Error('没有对应的 apikeys 配置，请再试一次。')
  }

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
        }
        catch (e) {
          console.error(`Error reading text file ${fileKey}`, e)
          fileContext += `\n\n[System Error: File ${fileKey} not found or unreadable]\n`
        }
      }
    }

    const finalMessage = message + (fileContext ? `\n\nAttached Files content:\n${fileContext}` : '')

    if (imageFiles.length > 0) {
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
    // ① Gemini
    if (isGeminiImageModel) {
      const baseUrl = isNotEmptyString(key.baseUrl) ? key.baseUrl.replace(/\/+$/, '') : undefined
      const { metas, allUrls } = await buildGeminiHistoryFromLastMessageId({
        lastMessageId: lastContext.parentMessageId,
        maxContextCount,
        forGeminiImageModel: true,
      })
      const { cleanedText: currentCleanedText, urls: currentUrls } = await extractImageUrlsFromMessageContent(content)
      const bindings = selectTwoImages(metas, allUrls, currentUrls)
      const history = metas.map(m => {
        const labeled = applyImageBindingsToCleanedText(m.cleanedText, m.urls, bindings)
        return { role: m.role, parts: [{ text: labeled || '[Empty]' }] }
      })

      const inputParts: GeminiPart[] = []
      if (bindings.length) {
        const mapText = bindings.map(b => `${b.tag}=${b.kind === 'initial' ? '最初参考图' : '上一轮生成图'}(${b.url})`).join('\n')
        inputParts.push({ text: `【图片映射表】\n${mapText}\n` })
      }
      for (const b of bindings) {
        inputParts.push({ text: `【${b.tag}：${b.kind === 'initial' ? '最初参考图' : '上一轮生成图'}】` })
        const imgPart = await imageUrlToGeminiInlinePart(b.url)
        if (imgPart) inputParts.push(imgPart)
      }
      const userInstruction = (currentCleanedText && currentCleanedText.trim()) ? currentCleanedText.trim() : '请继续生成/修改图片。'
      inputParts.push({ text: `【编辑指令】${userInstruction}` })

      const ai = new GoogleGenAI({
        apiKey: key.key,
        ...(baseUrl ? { httpOptions: { baseUrl } } : {}),
      })

      const contents = [
        ...history,
        { role: 'user', parts: inputParts },
      ]

      const response = await ai.models.generateContent({
        model,
        contents,
        config: {
          responseModalities: ['TEXT', 'IMAGE'],
          imageConfig: { imageSize: '4K' },
          ...(systemMessage ? { systemInstruction: systemMessage } as any : {}),
        } as any,
      } as any)

      if (!response.candidates || response.candidates.length === 0) {
        throw new Error(`[Gemini] Empty candidates.`)
      }

      let text = ''
      const parts = response.candidates?.[0]?.content?.parts ?? []

      for (const part of parts as any[]) {
        if (part?.text) {
          const rawText = part.text as string
          const replaced = await replaceDataUrlImagesWithUploads(rawText)
          text += replaced.text
        }
        const inline = part?.inlineData
        const inlineB64 = inline?.data
        if (inlineB64) {
          // save image...
          const mime = inline.mimeType || 'image/png'
          const base64 = inlineB64 as string
          const buffer = Buffer.from(base64, 'base64')
          const ext = mime.split('/')[1] || 'png'
          const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
          const filePath = path.join(UPLOAD_DIR, filename)
          await fs.writeFile(filePath, buffer)
          const prefix = text ? '\n\n' : ''
          text += `${prefix}![Generated Image](/uploads/${filename})`
        }
      }

      if (!text) text = '[Gemini] Success but no text/image parts returned.'

      const usageRes: any = (response as any).usageMetadata
        ? {
            prompt_tokens: (response as any).usageMetadata.promptTokenCount ?? 0,
            completion_tokens: (response as any).usageMetadata.candidatesTokenCount ?? 0,
            total_tokens: (response as any).usageMetadata.totalTokenCount ?? 0,
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

    // ② OpenAI
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
    let usageRes: any

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
      // Stream
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
    if (code === 429 && (error.message.includes('Too Many Requests') || error.message.includes('Rate limit'))) {
      if (options.tryCount++ < 3) {
        _lockedKeys.push({ key: key.key, lockedTime: Date.now() })
        await new Promise(resolve => setTimeout(resolve, 2000))
        const index = processThreads.findIndex(d => d.key === threadKey)
        if (index > -1) processThreads.splice(index, 1)
        return await chatReplyProcess(options)
      }
    }
    globalThis.console.error(error)
    if (Reflect.has(ErrorCodeMessage, code))
      return sendResponse({ type: 'Fail', message: ErrorCodeMessage[code] })
    return sendResponse({ type: 'Fail', message: error.message ?? 'Please check the back-end console' })
  }
  finally {
    const index = processThreads.findIndex(d => d.key === threadKey)
    if (index > -1) processThreads.splice(index, 1)
  }
}

// ✅ 修改：支持 roomId
export function abortChatProcess(userId: string, roomId: number) {
  const key = makeThreadKey(userId, roomId)
  const index = processThreads.findIndex(d => d.key === key)
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
            }
            catch (e) { }
          }
          promptText += (fileContext ? `\n\n[Attached Files History]:\n${fileContext}` : '')
        }

        if (promptText && typeof promptText === 'string') {
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
      else {
        let responseText = chatInfo.response || ''
        if (responseText && typeof responseText === 'string') {
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
    }
  }
  else {
    return undefined
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

// ✅ 修改：状态支持 Key
export function getChatProcessState(userId: string) {
  // 原有的单用户查询在 Job 模式下已主要由路由层的 jobStates 接管，但 retain 兼容
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
