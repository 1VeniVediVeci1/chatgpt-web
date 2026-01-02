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

// ===================== Gemini 图片模型 Debug 工具 =====================
// 你使用 docker-compose，环境变量为 DEBUG_GEMINI_IMAGE
function isGeminiImageDebugEnabled() {
  return process.env.DEBUG_GEMINI_IMAGE === 'true'
}

function debugMaxText() {
  const n = Number(process.env.DEBUG_GEMINI_IMAGE_MAX_TEXT ?? 800)
  return Number.isFinite(n) ? Math.max(100, n) : 800
}

function safeTruncate(str: string, max = debugMaxText()) {
  if (!str) return ''
  if (str.length <= max) return str
  return `${str.slice(0, max)}...[truncated ${str.length - max}]`
}

function safeBase64Preview(b64: string, head = 24, tail = 24) {
  if (!b64) return { len: 0, head: '', tail: '' }
  return {
    len: b64.length,
    head: b64.slice(0, head),
    tail: b64.slice(-tail),
  }
}

function debugPartSummary(part: any) {
  const summary: any = {}
  if (part?.text) {
    summary.type = 'text'
    summary.textLen = part.text.length
    summary.textPreview = safeTruncate(part.text)
  }
  else if (part?.inlineData?.data) {
    summary.type = 'inlineData'
    summary.mimeType = part.inlineData.mimeType
    summary.data = safeBase64Preview(part.inlineData.data)
  }
  else if (part?.fileData) {
    summary.type = 'fileData'
    summary.fileData = part.fileData
  }
  else {
    summary.type = 'unknown'
    summary.keys = Object.keys(part || {})
  }
  return summary
}

function debugGeminiResponseShape(response: any) {
  const candidates = response?.candidates ?? []
  const first = candidates?.[0]
  const parts = first?.content?.parts ?? []
  return {
    topKeys: Object.keys(response || {}),
    candidatesCount: candidates.length,
    firstCandidateKeys: first ? Object.keys(first) : [],
    firstContentKeys: first?.content ? Object.keys(first.content) : [],
    partsCount: parts.length,
    parts: parts.map((p: any, idx: number) => ({ idx, ...debugPartSummary(p) })),
    usageMetadata: response?.usageMetadata ?? null,
    promptFeedback: response?.promptFeedback ?? null,
  }
}

function debugGeminiInputParts(parts: any[]) {
  const out: any[] = []
  for (const [i, p] of parts.entries()) {
    if (p?.text) {
      out.push({
        idx: i,
        type: 'text',
        textLen: p.text.length,
        textPreview: safeTruncate(p.text),
      })
    }
    else if (p?.inlineData?.data) {
      out.push({
        idx: i,
        type: 'inlineData',
        mimeType: p.inlineData.mimeType,
        data: safeBase64Preview(p.inlineData.data),
      })
    }
    else {
      out.push({ idx: i, type: 'unknown', keys: Object.keys(p || {}) })
    }
  }
  return out
}

function debugGeminiHistory(history: any[]) {
  return (history || []).map((m: any, idx: number) => {
    const role = m?.role
    const parts = m?.parts ?? []
    const texts = parts
      .filter((p: any) => p?.text)
      .map((p: any) => safeTruncate(p.text))
    const inlineCount = parts.filter((p: any) => p?.inlineData?.data).length
    return { idx, role, textParts: texts, inlineCount }
  })
}
// ====================================================================

// ====== 解析文本中的图片引用（避免 base64 作为文本进入 Gemini） ======
const DATA_URL_IMAGE_RE = /data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g

// 用于捕获 mime 与 base64，以便落盘
const DATA_URL_IMAGE_CAPTURE_RE = /data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)/g

// 更宽松：先抓出括号内整体内容，再手动解析 URL（兼容 title、换行、额外参数）
const MARKDOWN_IMAGE_RE = /!$$[^$$]*]$\s*([\s\S]*?)\s*$/g

// 兜底：直接抓 /uploads/...（防止某些奇怪 Markdown 没被上面捕获）
const UPLOADS_URL_RE = /(\/uploads\/[^)\s>"']+\.(?:png|jpe?g|webp|gif|bmp|heic))/gi

// html: <img src="...">
const HTML_IMAGE_RE = /<img[^>]*\ssrc=["']([^"']+)["'][^>]*>/gi

/**
 * 把 text 里出现的 data:image/...;base64,... 全部落盘到 /uploads
 * 并把 dataURL 替换为 /uploads/xxx.ext
 *
 * 这样 DB 里会存 ![image](/uploads/xxx.png) 而不是超长 base64
 */
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

    // 仅替换 dataURL 本体，Markdown 的 ![](...) 结构会保持
    out += `/uploads/${filename}`

    lastIndex = idx + full.length
  }

  out += text.slice(lastIndex)
  return { text: out, saved }
}

/**
 * 从一段文本中抽取所有图片 URL，并把图片位置替换成占位符 [Image]
 * - 支持 markdown 图片、HTML img、裸 data:image base64
 */
function extractImageUrlsFromText(text: string): { cleanedText: string; urls: string[] } {
  if (!text) return { cleanedText: '', urls: [] }

  const urls: string[] = []
  let cleaned = text

  // 1) markdown: ![alt](...anything...)
  // 拿到括号内整体内容后：
  // - 去掉首尾尖括号 <...>
  // - 取第一个 token 作为 URL（避免 title/参数影响）
  cleaned = cleaned.replace(MARKDOWN_IMAGE_RE, (_m, inside) => {
    const raw = String(inside ?? '').trim()
    if (raw) {
      // 例如：/uploads/a.png "title"  或  <...> "title"
      const firstToken = raw.split(/\s+/)[0] || ''
      const url = firstToken.replace(/^<|>$/g, '').trim()
      if (url) urls.push(url)
    }
    return '[Image]'
  })

  // 2) html: <img src="...">
  cleaned = cleaned.replace(HTML_IMAGE_RE, (_m, url) => {
    if (url) urls.push(url)
    return '[Image]'
  })

  // 3) 裸 data url
  const rawDataUrls = cleaned.match(DATA_URL_IMAGE_RE)
  if (rawDataUrls?.length) urls.push(...rawDataUrls)
  cleaned = cleaned.replace(DATA_URL_IMAGE_RE, '[Image]')

  // 4) 兜底：直接匹配 /uploads/xxx.(png|jpg|...)
  const uploadUrls = cleaned.match(UPLOADS_URL_RE)
  if (uploadUrls?.length) urls.push(...uploadUrls)
  cleaned = cleaned.replace(UPLOADS_URL_RE, '[Image]')

  return { cleanedText: cleaned, urls }
}

/**
 * 从 MessageContent(可能是 string 或 OpenAI 多模态数组) 中：
 * 1) 抽取所有图片 url
 * 2) 返回“清理后的文本”(图片位置替换成 [Image])
 */
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

  return { cleanedText: '', urls }
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
  // data url
  if (urlStr.startsWith('data:')) {
    const [header, base64Data] = urlStr.split(',')
    if (!header || !base64Data) return null
    const mimeMatch = header.match(/:(.*?);/)
    const mimeType = mimeMatch ? mimeMatch[1] : 'image/png'
    return { inlineData: { mimeType, data: base64Data } }
  }

  // /uploads/xxx
  if (urlStr.includes('/uploads/')) {
    const filename = path.basename(urlStr)
    const fileData = await getFileBase64(filename)
    if (!fileData) return null
    return { inlineData: { mimeType: fileData.mime, data: fileData.data } }
  }

  return null
}

async function messageContentToGeminiParts(content: MessageContent, forGeminiImageModel: boolean): Promise<Part[]> {
  const parts: Part[] = []

  // 1) 纯字符串（assistant 常见：markdown + 可能夹 dataURL）
  if (typeof content === 'string') {
    if (forGeminiImageModel) {
      const { cleanedText, urls } = extractImageUrlsFromText(content)
      if (cleanedText) parts.push({ text: cleanedText })

      for (const url of urls) {
        const imgPart = await imageUrlToGeminiInlinePart(url)
        if (imgPart) parts.push(imgPart)
      }
      return parts
    }

    parts.push({ text: content })
    return parts
  }

  // 2) OpenAI 多模态数组（user 常见：[{type:'text'}, {type:'image_url'}]）
  if (Array.isArray(content)) {
    for (const p of content as any[]) {
      if (p?.type === 'text' && p.text) {
        if (forGeminiImageModel) {
          const { cleanedText, urls } = extractImageUrlsFromText(p.text)
          if (cleanedText) parts.push({ text: cleanedText })

          for (const url of urls) {
            const imgPart = await imageUrlToGeminiInlinePart(url)
            if (imgPart) parts.push(imgPart)
          }
        }
        else {
          parts.push({ text: p.text })
        }
      }
      else if (p?.type === 'image_url' && p.image_url?.url) {
        const imgPart = await imageUrlToGeminiInlinePart(p.image_url.url)
        if (imgPart) parts.push(imgPart)
      }
    }
  }

  return parts
}

/**
 * 核心逻辑：按原有 DB 链条回溯，构造 Gemini 需要的 history
 * 但：history 中不再放任何 inlineData 图片，只保留文本占位符；
 * 同时把“上下文中出现过的所有图片 URL”收集出来，供最新一轮 inputParts 使用。
 */
async function buildGeminiHistoryFromLastMessageId(params: {
  lastMessageId?: string
  maxContextCount: number
  forGeminiImageModel: boolean
}): Promise<{ history: any[]; contextImageUrls: string[] }> {
  const { lastMessageId: startId, maxContextCount, forGeminiImageModel } = params

  const messages: { role: 'user' | 'model'; parts: Part[] }[] = []
  const collectedUrls: string[] = []

  let lastMessageId = startId
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId) break
    const msg = await getMessageById(lastMessageId)
    if (!msg) break

    // 系统内部 assistant 对应 Gemini model
    const role = (msg.role === 'assistant' ? 'model' : 'user') as 'user' | 'model'

    if (forGeminiImageModel) {
      // 1) 抽取图片 URL（包含 assistant 生成的 /uploads 与 user 上传 data url）
      const { cleanedText, urls } = await extractImageUrlsFromMessageContent(msg.text)
      if (urls.length) collectedUrls.push(...urls)

      // 2) history 里只放文本（图片位置用 [Image] 占位符）
      const safeText = (cleanedText && cleanedText.trim())
        ? cleanedText
        : (urls.length ? '[Image]' : '[Empty]')

      messages.push({ role, parts: [{ text: safeText }] })
    }
    else {
      // 非图片模型：保留原逻辑
      const parts = await messageContentToGeminiParts(msg.text, false)
      messages.push({ role, parts })
    }

    lastMessageId = msg.parentMessageId
  }

  messages.reverse()
  collectedUrls.reverse()

  return {
    history: messages.map(m => ({ role: m.role, parts: m.parts })),
    contextImageUrls: dedupeUrlsPreserveOrder(collectedUrls),
  }
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
    // 防止把 data:image;base64 巨长字符串塞回上下文
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

  // Gemini 图片模型直接走下方特殊逻辑，这里主要是 Other/Image(DallE)
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
      console.log(`[OpenAI] reasoning_effort enabled: model=${model}, value=${reasoningEffort}`)
    }
  }
  catch (e) {
    console.error('[OpenAI] set reasoning_effort failed:', e)
  }

  console.log(`[OpenAI] Request Model: ${model}, URL: ${OPENAI_API_BASE_URL}`)

  const apiResponse = await openai.chat.completions.create(options, {
    signal: abortSignal,
  })

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
  if (existingIdx > -1) {
    processThreads.splice(existingIdx, 1)
  }

  console.log(`[DEBUG] Pushing thread for userId: ${userId}, roomId: ${roomId}`)
  processThreads.push({ userId, abort, messageId, roomId })

  await ensureUploadDir()
  const model = options.room.chatModel
  const key = await getRandomApiKey(options.user, model, options.room.accountId)
  const maxContextCount = options.user.advanced.maxContextCount ?? 20
  if (key == null || key === undefined) {
    const idx = processThreads.findIndex(d => d.userId === userId)
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
    }
    else {
      content = finalMessage
    }
  }

  try {
    // ====== ① Gemini 图片模型（上下文所有图放进最新一轮 inputParts；history 只保留占位符） ======
    if (isGeminiImageModel) {
      const baseUrl = isNotEmptyString(key.baseUrl) ? key.baseUrl : undefined

      // 1) 构造历史（history 仅文本，占位符保留；同时收集上下文图片 URLs）
      const { history, contextImageUrls } = await buildGeminiHistoryFromLastMessageId({
        lastMessageId: lastContext.parentMessageId,
        maxContextCount,
        forGeminiImageModel: true,
      })

      // 2) 抽取本轮输入：文本(保留占位符) + 本轮图片 URLs
      const { cleanedText: currentCleanedText, urls: currentUrls } = await extractImageUrlsFromMessageContent(content)

      // 3) 合并“上下文所有图 + 本轮所有图”，去重
      const allImageUrls = dedupeUrlsPreserveOrder([
        ...contextImageUrls,
        ...currentUrls,
      ])

      // 4) 构造 inputParts：把所有图都塞进最新一轮，再塞文本指令
      const inputParts: Part[] = []

      for (const u of allImageUrls) {
        const imgPart = await imageUrlToGeminiInlinePart(u)
        if (imgPart) inputParts.push(imgPart)
      }

      // 文字必须有
      const finalPromptText = (currentCleanedText && currentCleanedText.trim())
        ? currentCleanedText.trim()
        : (inputParts.length ? '请基于以上所有图片继续生成/修改。' : '请根据要求生成图片。')

      inputParts.push({ text: finalPromptText })

      // 5) 创建模型实例并 startChat
      const genAI = new GoogleGenerativeAI(key.key)
      const requestOptions: any = {}
      if (globalConfig.timeoutMs) requestOptions.timeout = globalConfig.timeoutMs
      if (baseUrl) requestOptions.baseUrl = baseUrl.replace(/\/+$/, '')

      const geminiModel = genAI.getGenerativeModel({
        model,
        systemInstruction: systemMessage ? { parts: [{ text: systemMessage }], role: 'system' } : undefined,
      }, requestOptions)

      const chatSession = geminiModel.startChat({ history })

      // ===== DEBUG: 请求前输出 =====
      if (isGeminiImageDebugEnabled()) {
        console.log('[GeminiImage DEBUG] model=', model)
        console.log('[GeminiImage DEBUG] history=', JSON.stringify(debugGeminiHistory(history), null, 2))
        console.log('[GeminiImage DEBUG] contextImageUrls=', contextImageUrls)
        console.log('[GeminiImage DEBUG] currentUrls=', currentUrls)
        console.log('[GeminiImage DEBUG] allImageUrls=', allImageUrls)
        console.log('[GeminiImage DEBUG] inputParts=', JSON.stringify(debugGeminiInputParts(inputParts as any), null, 2))
      }

      const result = await chatSession.sendMessage(inputParts)
      const response = await result.response

      // ===== DEBUG: 返回结构输出 =====
      if (isGeminiImageDebugEnabled()) {
        console.log('[GeminiImage DEBUG] responseShape=', JSON.stringify(debugGeminiResponseShape(response), null, 2))
      }

      if (!response.candidates || response.candidates.length === 0) {
        throw new Error(`[Gemini] Empty candidates. usage=${JSON.stringify(response.usageMetadata ?? {})}`)
      }

      let text = ''
      const parts = response.candidates?.[0]?.content?.parts ?? []

      for (const part of parts as any[]) {
        // ① 先处理 text：Gemini 有时把图片用 Markdown(dataURL) 塞进 text
        if (part?.text) {
          const rawText = part.text as string
          const replaced = await replaceDataUrlImagesWithUploads(rawText)

          if (isGeminiImageDebugEnabled() && replaced.saved.length) {
            console.log('[GeminiImage DEBUG] extractedDataUrlImagesFromText=', replaced.saved.map(s => ({
              mime: s.mime,
              filename: s.filename,
              bytes: s.bytes,
            })))
          }

          text += replaced.text
        }

        // ② 再处理 inlineData：兼容某些模型确实会用 inlineData 返回图片
        if (part?.inlineData?.data) {
          const mime = part.inlineData.mimeType || 'image/png'
          const base64 = part.inlineData.data as string
          const buffer = Buffer.from(base64, 'base64')
          const ext = mime.split('/')[1] || 'png'
          const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
          const filePath = path.join(UPLOAD_DIR, filename)
          await fs.writeFile(filePath, buffer)

          if (isGeminiImageDebugEnabled()) {
            console.log('[GeminiImage DEBUG] savedImageFromInlineData=', {
              mime,
              bytes: buffer.length,
              filename,
              filePath,
              base64: safeBase64Preview(base64),
            })
          }

          const prefix = text ? '\n\n' : ''
          text += `${prefix}![Generated Image](/uploads/${filename})`
        }
      }

      if (!text) text = '[Gemini] Success but no text/image parts returned.'

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

    // ====== ② 其它模型 (OpenAI / Dall-E / Normal Gemini) ======
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
    if (code === 429 && (error.message.includes('Too Many Requests') || error.message.includes('Rate limit'))) {
      if (options.tryCount++ < 3) {
        _lockedKeys.push({ key: key.key, lockedTime: Date.now() })
        await new Promise(resolve => setTimeout(resolve, 2000))
        const index = processThreads.findIndex(d => d.userId === userId)
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

        // 防止把 data:image;base64 巨长字符串塞回上下文
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
        // 去除冗余的base64 防止加载过慢/上下文爆炸
        // 注意：我们在生成阶段已经会把 dataURL 落盘成 /uploads，
        // 所以这里替换主要是兜底处理旧数据/异常数据。
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
