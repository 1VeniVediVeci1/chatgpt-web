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

// ====== 图片正则 / 占位符体系（Image#n）+ 映射表 ======
type ImageRef = {
  tag: string // Image#1
  placeholder: string // [Image#1]
  url: string // /uploads/... or data:image...
}
type ImageMapItem = ImageRef & {
  role: 'user' | 'model'
  sourceMessageId: string
  turnIndex: number // history 中第几条消息（从 0 开始）
}

function makeImageTag(n: number) {
  return `Image#${n}`
}

function makePlaceholder(tag: string) {
  return `[${tag}]`
}

function isLikelyImageUrl(u: string) {
  if (!u) return false
  if (u.startsWith('data:image/')) return true
  if (u.includes('/uploads/')) return true
  if (/^https?:\/\//i.test(u)) return true
  return false
}

// ====== 解析文本中的图片引用（避免 base64 作为文本进入 Gemini） ======
const DATA_URL_IMAGE_RE = /data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g
const DATA_URL_IMAGE_CAPTURE_RE = /data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)/g

// 更宽松：抓取 ![...](...) 括号内整体内容
const MARKDOWN_IMAGE_RE = /!$$[^$$]*]$\s*([\s\S]*?)\s*$/g

// html: <img src="...">
const HTML_IMAGE_RE = /<img[^>]*\ssrc=["']([^"']+)["'][^>]*>/gi

// 兜底：直接抓 /uploads/xxx.(png|jpg|...)
const UPLOADS_URL_RE = /(\/uploads\/[^)\s>"']+\.(?:png|jpe?g|webp|gif|bmp|heic))/gi

/**
 * 把 text 里出现的 data:image/...;base64,... 全部落盘到 /uploads
 * 并把 dataURL 替换为 /uploads/xxx.ext
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

    out += `/uploads/${filename}`

    lastIndex = idx + full.length
  }

  out += text.slice(lastIndex)
  return { text: out, saved }
}

/**
 * 从文本中抽取图片，并替换为 [Image#n] 占位符，同时返回映射
 */
function extractImageRefsFromTextWithNumbering(text: string, nextIndexRef: { value: number }): { cleanedText: string; images: ImageRef[] } {
  if (!text) return { cleanedText: '', images: [] }

  const images: ImageRef[] = []
  let cleaned = text

  // 1) Markdown 图片：![alt](...anything...)
  cleaned = cleaned.replace(MARKDOWN_IMAGE_RE, (_m, inside) => {
    const raw = String(inside ?? '').trim()
    const firstToken = raw.split(/\s+/)[0] || ''
    const url = firstToken.replace(/^<|>$/g, '').trim()
    // 只有像图片 URL 才入表；否则只是占位符/垃圾内容就不入表，但仍替换掉
    nextIndexRef.value += 1
    const tag = makeImageTag(nextIndexRef.value)
    const placeholder = makePlaceholder(tag)
    if (isLikelyImageUrl(url)) images.push({ tag, placeholder, url })
    return placeholder
  })

  // 2) HTML img
  cleaned = cleaned.replace(HTML_IMAGE_RE, (_m, url) => {
    nextIndexRef.value += 1
    const tag = makeImageTag(nextIndexRef.value)
    const placeholder = makePlaceholder(tag)
    if (isLikelyImageUrl(url)) images.push({ tag, placeholder, url })
    return placeholder
  })

  // 3) 裸 data url
  cleaned = cleaned.replace(DATA_URL_IMAGE_RE, (m) => {
    nextIndexRef.value += 1
    const tag = makeImageTag(nextIndexRef.value)
    const placeholder = makePlaceholder(tag)
    images.push({ tag, placeholder, url: m })
    return placeholder
  })

  // 4) 裸 /uploads 兜底
  cleaned = cleaned.replace(UPLOADS_URL_RE, (m) => {
    nextIndexRef.value += 1
    const tag = makeImageTag(nextIndexRef.value)
    const placeholder = makePlaceholder(tag)
    images.push({ tag, placeholder, url: m })
    return placeholder
  })

  return { cleanedText: cleaned, images }
}

/**
 * 从 MessageContent 中抽取图片，并替换为 [Image#n]
 * - string: 直接处理
 * - array: 对 text 部分处理；对 image_url 部分追加一个占位符到文本末尾
 */
async function extractImageRefsFromMessageContentWithNumbering(
  content: MessageContent,
  nextIndexRef: { value: number },
): Promise<{ cleanedText: string; images: ImageRef[] }> {
  const images: ImageRef[] = []
  let cleanedText = ''

  if (typeof content === 'string') {
    const r = extractImageRefsFromTextWithNumbering(content, nextIndexRef)
    return { cleanedText: r.cleanedText, images: r.images }
  }

  if (Array.isArray(content)) {
    for (const p of content as any[]) {
      if (p?.type === 'text' && p.text) {
        const r = extractImageRefsFromTextWithNumbering(p.text, nextIndexRef)
        cleanedText += (cleanedText ? '\n' : '') + r.cleanedText
        images.push(...r.images)
      }
      else if (p?.type === 'image_url' && p.image_url?.url) {
        nextIndexRef.value += 1
        const tag = makeImageTag(nextIndexRef.value)
        const placeholder = makePlaceholder(tag)
        images.push({ tag, placeholder, url: p.image_url.url })
        cleanedText += (cleanedText ? '\n' : '') + placeholder
      }
    }
    if (!cleanedText?.trim() && images.length) {
      cleanedText = images.map(i => i.placeholder).join(' ')
    }
    return { cleanedText, images }
  }

  return { cleanedText: '', images: [] }
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

  if (typeof content === 'string') {
    parts.push({ text: content })
    return parts
  }

  if (Array.isArray(content)) {
    for (const p of content as any[]) {
      if (p?.type === 'text' && p.text) {
        parts.push({ text: p.text })
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
 * buildGeminiHistoryFromLastMessageId：
 * - history: 只放文本（含 [Image#n] 占位符）
 * - imageMap: 占位符 Image#n -> url 的映射（带元信息）
 * - nextImageIndex: 下一个可用编号
 */
async function buildGeminiHistoryFromLastMessageId(params: {
  lastMessageId?: string
  maxContextCount: number
  forGeminiImageModel: boolean
}): Promise<{ history: any[]; imageMap: ImageMapItem[]; nextImageIndex: number }> {
  const { lastMessageId: startId, maxContextCount, forGeminiImageModel } = params

  const rawMsgs: Array<{ id: string; role: 'user' | 'model'; content: MessageContent }> = []

  let lastMessageId = startId
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId) break
    const msg = await getMessageById(lastMessageId)
    if (!msg) break

    const role = (msg.role === 'assistant' ? 'model' : 'user') as 'user' | 'model'
    rawMsgs.push({ id: msg.id, role, content: msg.text })

    lastMessageId = msg.parentMessageId
  }

  rawMsgs.reverse()

  const nextIndexRef = { value: 0 }
  const history: any[] = []
  const imageMap: ImageMapItem[] = []

  for (let turn = 0; turn < rawMsgs.length; turn++) {
    const m = rawMsgs[turn]

    if (forGeminiImageModel) {
      const { cleanedText, images } = await extractImageRefsFromMessageContentWithNumbering(m.content, nextIndexRef)

      // 记录映射表（只记录像图片 URL 的项）
      for (const img of images) {
        if (!isLikelyImageUrl(img.url)) continue
        imageMap.push({
          ...img,
          role: m.role,
          sourceMessageId: m.id,
          turnIndex: turn,
        })
      }

      const safeText = (cleanedText && cleanedText.trim())
        ? cleanedText
        : (images.length ? images.map(i => i.placeholder).join(' ') : '[Empty]')

      history.push({ role: m.role, parts: [{ text: safeText }] })
    }
    else {
      const parts = await messageContentToGeminiParts(m.content, false)
      history.push({ role: m.role, parts })
    }
  }

  return { history, imageMap, nextImageIndex: nextIndexRef.value }
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
      console.log(`[OpenAI] reasoning_effort enabled: model=${model}, value=${reasoningEffort}`)
    }
  }
  catch (e) {
    console.error('[OpenAI] set reasoning_effort failed:', e)
  }

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
  if (key == null) {
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
    // =================== Gemini 图片模型（Image#n 映射表版） ===================
    if (isGeminiImageModel) {
      const baseUrl = isNotEmptyString(key.baseUrl) ? key.baseUrl : undefined

      // 1) 生成 history + imageMap（历史中的图片变成 [Image#n]）
      const hist = await buildGeminiHistoryFromLastMessageId({
        lastMessageId: lastContext.parentMessageId,
        maxContextCount,
        forGeminiImageModel: true,
      })
      const history = hist.history
      const historyImageMap = hist.imageMap

      // 2) 当前轮：继续编号（从 nextImageIndex 开始）
      const nextIndexRef = { value: hist.nextImageIndex }
      const currentExtract = await extractImageRefsFromMessageContentWithNumbering(content, nextIndexRef)

      // 当前轮图片也纳入映射表（这里 role 固定为 user；sourceMessageId 用本轮 messageId/自定义）
      const currentImageMap: ImageMapItem[] = currentExtract.images
        .filter(i => isLikelyImageUrl(i.url))
        .map((img) => ({
          ...img,
          role: 'user',
          sourceMessageId: `current_${customMessageId}`,
          turnIndex: history.length, // 当前轮在 history 之后
        }))

      const allImageMap: ImageMapItem[] = [...historyImageMap, ...currentImageMap]

      const latest = allImageMap.length ? allImageMap[allImageMap.length - 1] : null

      // 3) 构造 inputParts：
      //    - 先发一段映射表说明文本
      //    - 然后每张图：文本标签（含 Image#n） + inlineData
      //    - 最后：指令文本（尽量引用最新 Image#n）
      const inputParts: Part[] = []

      if (allImageMap.length) {
        const mapLines = allImageMap.map((img) => {
          const who = img.role === 'model' ? '模型生成' : '用户提供'
          return `- ${img.tag} (${img.placeholder})：${who}，sourceMessageId=${img.sourceMessageId}，turn=${img.turnIndex}`
        }).join('\n')

        inputParts.push({
          text:
            `【图片映射表】（占位符 -> 图片）\n${mapLines}\n\n` +
            `接下来将按映射表顺序依次提供对应图片内容：`,
        })

        for (const img of allImageMap) {
          inputParts.push({ text: `【${img.tag}】对应占位符 ${img.placeholder}` })
          const imgPart = await imageUrlToGeminiInlinePart(img.url)
          if (imgPart) inputParts.push(imgPart)
          else inputParts.push({ text: `【警告】${img.tag} 的图片 URL 无法读取为 inlineData：${img.url}` })
        }
      }

      // 4) 最终指令：如果用户没写 Image#，默认基于最新图
      let userInstruction = (currentExtract.cleanedText || '').trim()
      if (!userInstruction) {
        userInstruction = allImageMap.length
          ? `请基于最新图片 ${latest?.placeholder} 继续生成/修改。`
          : '请根据要求生成图片。'
      }

      const hasExplicitRef = /Image#\d+/.test(userInstruction)
      if (!hasExplicitRef && latest) {
        userInstruction = `请基于【${latest.tag}】(${latest.placeholder})进行修改：${userInstruction}`
      }

      inputParts.push({ text: `【当前指令】${userInstruction}` })

      // 5) 创建模型并 startChat
      const genAI = new GoogleGenerativeAI(key.key)
      const requestOptions: any = {}
      if (globalConfig.timeoutMs) requestOptions.timeout = globalConfig.timeoutMs
      if (baseUrl) requestOptions.baseUrl = baseUrl.replace(/\/+$/, '')

      const geminiModel = genAI.getGenerativeModel({
        model,
        systemInstruction: systemMessage ? { parts: [{ text: systemMessage }], role: 'system' } : undefined,
      }, requestOptions)

      const chatSession = geminiModel.startChat({ history })

      // DEBUG：请求前输出
      if (isGeminiImageDebugEnabled()) {
        console.log('[GeminiImage DEBUG] model=', model)
        console.log('[GeminiImage DEBUG] history=', JSON.stringify(debugGeminiHistory(history), null, 2))
        console.log('[GeminiImage DEBUG] imageMap=', allImageMap.map(i => ({
          tag: i.tag,
          placeholder: i.placeholder,
          url: i.url,
          role: i.role,
          sourceMessageId: i.sourceMessageId,
          turnIndex: i.turnIndex,
        })))
        console.log('[GeminiImage DEBUG] inputParts=', JSON.stringify(debugGeminiInputParts(inputParts as any), null, 2))
      }

      const result = await chatSession.sendMessage(inputParts)
      const response = await result.response

      if (isGeminiImageDebugEnabled()) {
        console.log('[GeminiImage DEBUG] responseShape=', JSON.stringify(debugGeminiResponseShape(response), null, 2))
      }

      if (!response.candidates || response.candidates.length === 0) {
        throw new Error(`[Gemini] Empty candidates. usage=${JSON.stringify(response.usageMetadata ?? {})}`)
      }

      // 6) 解析返回：Gemini-3-pro-image 常把图片塞在 part.text 的 dataURL
      let text = ''
      const parts = response.candidates?.[0]?.content?.parts ?? []

      for (const part of parts as any[]) {
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

        // 兼容 inlineData
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

    // =================== 其它模型 (OpenAI / Dall-E / Normal Gemini) ===================
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
        // 兜底：旧数据可能仍含 dataURL，避免爆上下文
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
