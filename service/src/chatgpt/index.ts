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
import { abortablePromise, isAbortError } from './abortable'
import { webSearch } from '../utils/webSearch'

type GeminiPart = { text?: string; inlineData?: { mimeType: string; data: string } }

const MODEL_CONFIGS: Record<string, { supportTopP: boolean; defaultTemperature?: number }> = {
  'gpt-5-search-api': { supportTopP: false, defaultTemperature: 0.8 },
}

const UPLOAD_DIR = path.resolve(process.cwd(), 'uploads')

async function ensureUploadDir() { try { await fs.access(UPLOAD_DIR) } catch { await fs.mkdir(UPLOAD_DIR, { recursive: true }) } }
function stripTypePrefix(key: string): string { return key.replace(/^(img:|txt:)/, '') }
function isTextFile(filename: string): boolean {
  if (filename.startsWith('txt:')) return true; if (filename.startsWith('img:')) return false
  const ext = path.extname(stripTypePrefix(filename)).toLowerCase()
  return ['.txt', '.md', '.json', '.csv', '.js', '.ts', '.py', '.java', '.html', '.css', '.xml', '.yml', '.yaml', '.log', '.ini', '.config'].includes(ext)
}
function isImageFile(filename: string): boolean {
  if (filename.startsWith('img:')) return true; if (filename.startsWith('txt:')) return false
  const ext = path.extname(stripTypePrefix(filename)).toLowerCase()
  return ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.heic', '.bmp'].includes(ext)
}
async function getFileBase64(filename: string): Promise<{ mime: string; data: string } | null> {
  try {
    const realFilename = stripTypePrefix(filename); const filePath = path.join(UPLOAD_DIR, realFilename)
    await fs.access(filePath); const buffer = await fs.readFile(filePath)
    const ext = path.extname(realFilename).toLowerCase().replace('.', '')
    let mime = 'image/png'
    if (ext === 'jpg' || ext === 'jpeg') mime = 'image/jpeg'
    else if (ext === 'webp') mime = 'image/webp'
    else if (ext === 'gif') mime = 'image/gif'
    else if (ext === 'heic') mime = 'image/heic'
    else if (ext === 'bmp') mime = 'image/bmp'
    return { mime, data: buffer.toString('base64') }
  } catch (e) { globalThis.console.error(`[File Read Error] ${filename}:`, e); return null }
}

dotenv.config()

const API_DEBUG = process.env.API_DEBUG === 'true'
const API_DEBUG_MAX_TEXT = Number(process.env.API_DEBUG_MAX_TEXT ?? 800)

function trunc(s: any, n = API_DEBUG_MAX_TEXT): string { const str = typeof s === 'string' ? s : JSON.stringify(s ?? ''); if (!str) return ''; return str.length > n ? `${str.slice(0, n)}...(truncated,total=${str.length})` : str }
function safeJson(obj: any): string { try { return JSON.stringify(obj, null, 2) } catch (e: any) { return `[Unserializable: ${e?.message ?? e}]` } }
function summarizeOpenAIMessages(messages: any[]) {
  return (messages ?? []).map((m, idx) => {
    const content = m?.content; let contentType: string = typeof content; let contentLen = 0; let contentPreview: string | undefined
    if (typeof content === 'string') { contentLen = content.length; contentPreview = trunc(content) }
    else if (Array.isArray(content)) { contentType = 'array'; contentLen = content.length; contentPreview = trunc(content.map((p: any) => ({ type: p?.type, textLen: typeof p?.text === 'string' ? p.text.length : undefined, hasImageUrl: !!p?.image_url?.url }))) }
    else if (content === null) { contentType = 'null'; contentPreview = 'null' }
    else { contentType = 'object'; contentPreview = trunc(content) }
    return { idx, role: m?.role, contentType, contentLen, contentPreview }
  })
}
function debugLog(...args: any[]) { if (!API_DEBUG) return; console.log(...args) }

// ===================== 联网搜索 =====================
type SearchPlan = {
  action: 'search' | 'stop'
  query?: string
  reason?: string
  selected_ids?: string[] // e.g. ["1.1", "2.3"]
  context_summary?: string // 对上下文的总结
}
type SearchRound = { query: string; items: Array<{ title: string; url: string; content: string }>; note?: string }

function safeParseJsonFromText(text: string): any | null {
  if (!text) return null; const s = String(text).trim()
  try { return JSON.parse(s) } catch { }
  const i = s.indexOf('{'); const j = s.lastIndexOf('}')
  if (i >= 0 && j > i) { try { return JSON.parse(s.slice(i, j + 1)) } catch { } }
  return null
}

async function buildConversationContext(lastMessageId: string | undefined, maxCount: number): Promise<string> {
  if (!lastMessageId) return ''
  const messages: string[] = []
  let currentId = lastMessageId
  for (let i = 0; i < maxCount; i++) {
    if (!currentId) break
    const msg = await getMessageById(currentId)
    if (!msg) break
    const role = msg.role === 'assistant' ? 'AI' : 'User'
    let content = ''
    if (typeof msg.text === 'string') content = msg.text
    else if (Array.isArray(msg.text)) content = msg.text.map((p: any) => p?.type === 'text' ? p.text : '[Image]').join('')
    else content = '[Complex Content]'
    messages.push(`${role}: ${content}`)
    currentId = msg.parentMessageId
  }
  return messages.reverse().join('\n')
}

// 修改：展示全部条目，不做 slice(0,5)，确保模型看到完整结果来判断完整性
function formatSearchRoundsForPlanner(rounds: SearchRound[]): string {
  if (!rounds.length) return '（无）'
  return rounds.map((r, idx) => {
    // 这里的 content 保留完整文本，只做基本清洗
    const items = (r.items || []).map((it, i) =>
      `- [${idx + 1}.${i + 1}] ${String(it.title || '').trim()}\n  ${String(it.url || '').trim()}\n  内容: ${String(it.content || '').replace(/\s+/g, ' ').trim()}`
    ).join('\n\n')
    const note = r.note ? `\n（注：${r.note}）` : ''
    // [1.1] means Round 1 Item 1
    return `### 第${idx + 1}轮 query="${r.query}"\n${items || '（无结果）'}${note}`
  }).join('\n\n')
}

async function planNextSearchAction(params: {
  openai: OpenAI
  model: string
  userQuestion: string
  rounds: SearchRound[]
  fullContext: string
  priorContextSummary: string | null
  date: string
  abortSignal?: AbortSignal
}): Promise<SearchPlan> {
  const { openai, model, userQuestion, rounds, fullContext, priorContextSummary, date, abortSignal } = params

  const isFirstRound = !priorContextSummary 

  const plannerSystem = [
    '你是"联网搜索规划器 & 结果筛选器"。你的任务是协助回答用户问题。',
    '',
    `当前时间：${date}`,
    '',
    '任务：',
    '1. 首先判断是否需要联网搜索。**这是最重要的决策。**',
    '2. 评估"已进行的搜索与结果"，选出对回答问题有价值的条目ID (格式如 "1.1", "2.3")。',
    '',
    '【决策逻辑】',
    '- **必须搜索**：问题涉及近期新闻、特定事实查证（天气、股价等）、非常识性具体数据，且上下文中没有。',
    '- **禁止搜索 (action="stop")**：',
    '  - 纯闲聊（如“你好”、“你是谁”）。',
    '  - 常识性问题（如“太阳从哪升起”、“1+1等于几”）。',
    '  - 逻辑推理、数学计算、代码编写、翻译、润色任务。',
    '  - **上下文已包含答案**：用户问题是针对历史对话的追问，且历史信息已足够回答。',
    '',
    '如果决定搜索：action="search"，并给出 query。',
    '如果无需搜索或信息已足：action="stop"（此时 selected_ids 依然可以返回已有结果的ID，如果没有结果则为空数组）。',
    '',
    isFirstRound
      ? '3. 请务必阅读完整的对话上下文，并生成一个精炼的"context_summary"（100-200字），概括历史对话的核心意图和关键信息，以便后续轮次使用。'
      : '3. 参考提供的 "context_summary" 来理解用户意图（不再提供完整上下文）。',
    '',
    '输出严格 JSON：',
    '{',
    '  "action": "search" | "stop",',
    '  "query": string, // 如果 action=search',
    '  "reason": string,',
    '  "selected_ids": string[], // 选出的高质量结果ID列表，例如 ["1.1", "2.1"]',
    isFirstRound ? '  "context_summary": string // 对历史上下文的精炼总结' : null,
    '}'.replace(/, null/g, ''),
  ].filter(Boolean).join('\n')

  // 如果有之前的总结，就用总结；否则用完整上下文
  const contextBlock = isFirstRound
    ? `【完整对话上下文（请总结生成 context_summary）】\n${fullContext || '（无）'}`
    : `【历史上下文总结 (context_summary)】\n${priorContextSummary}`

  const plannerUser = [
    '【用户问题】', userQuestion,
    '',
    contextBlock,
    '',
    '【已进行的搜索与结果】',
    formatSearchRoundsForPlanner(rounds),
    '',
    '现在请决定：selected_ids 是哪些？是否需要继续搜索？' + (isFirstRound ? ' 记得生成 context_summary。' : '')
  ].join('\n')

  try {
    const resp = await openai.chat.completions.create({ model, temperature: 0, messages: [{ role: 'system', content: plannerSystem }, { role: 'user', content: plannerUser }], response_format: { type: 'json_object' } as any, stream: false } as any, { signal: abortSignal })
    const obj = safeParseJsonFromText(resp.choices?.[0]?.message?.content ?? '')
    return obj ? obj as SearchPlan : { action: 'stop', reason: 'planner json parse failed' }
  } catch {
    const resp = await openai.chat.completions.create({ model, temperature: 0, messages: [{ role: 'system', content: plannerSystem }, { role: 'user', content: `${plannerUser}\n\n只输出 JSON，不要输出其它文本。` }], stream: false } as any, { signal: abortSignal })
    const obj = safeParseJsonFromText(resp.choices?.[0]?.message?.content ?? '')
    return obj ? obj as SearchPlan : { action: 'stop', reason: 'planner json parse failed' }
  }
}

async function runIterativeWebSearch(params: {
  openai: OpenAI; plannerModels: string[]; userQuestion: string; maxRounds: number; maxResults: number
  abortSignal?: AbortSignal; provider?: 'searxng' | 'tavily'; searxngApiUrl?: string; tavilyApiKey?: string
  onProgress?: (status: string) => void
  fullContext: string; date: string
}): Promise<{ rounds: SearchRound[]; selectedIds: Set<string> }> {
  const { openai, plannerModels, userQuestion, maxRounds, maxResults, abortSignal, provider, searxngApiUrl, tavilyApiKey, onProgress, fullContext, date } = params
  const rounds: SearchRound[] = []
  const usedQueries = new Set<string>()
  const selectedIds = new Set<string>()
  
  // 存储第一轮生成的上下文总结，供后续使用
  let currentContextSummary: string | null = null

  for (let i = 0; i < maxRounds; i++) {
    // 状态更新：开始分析
    // 使用 emoji 和更自然的语言，减轻用户的等待焦虑
    onProgress?.(`⏳ 第 ${i + 1}/${maxRounds} 轮规划：正在仔细分析您的意图与上下文...`)
    
    let plan: SearchPlan | null = null
    for (const m of plannerModels) {
      try {
        plan = await planNextSearchAction({ 
          openai, 
          model: m, 
          userQuestion, 
          rounds, 
          abortSignal, 
          fullContext, 
          priorContextSummary: currentContextSummary, 
          date 
        })
        break 
      } catch (e) {
        if (API_DEBUG) debugLog('[SearchPlanner] model failed:', m, (e as any)?.message ?? e) 
      } 
    }
    
    if (!plan) { 
        onProgress?.('❌ 规划服务暂时不可用，流程结束。'); 
        break 
    }

    // 更新总结
    if (plan.context_summary && typeof plan.context_summary === 'string') {
      currentContextSummary = plan.context_summary
      if (API_DEBUG) debugLog('[WebSearch] Context Summary updated:', currentContextSummary)
    }

    // 累积选中的 ID
    if (Array.isArray(plan.selected_ids)) {
      plan.selected_ids.forEach(id => selectedIds.add(String(id).trim()))
    }

    // ✅ 如果第一轮 decide stop，说明不需要搜
    const reasonText = plan.reason ? `(理由: ${plan.reason})` : ''
    
    if (plan.action !== 'search') { 
        if (i === 0) onProgress?.(`🛑 模型判断无需搜索 ${reasonText}`)
        else onProgress?.(`✅ 信息收集完毕 ${reasonText}`)
        break
    }
    
    const q = String(plan.query || '').trim()
    if (!q) break; 
    
    if (usedQueries.has(q)) { 
        onProgress?.(`⚠️ 关键词「${q}」已使用过，跳过重复搜索...`); 
        break 
    }; 
    usedQueries.add(q)
    
    // 状态更新：开始搜索
    onProgress?.(`🔍 正在搜索：「${q}」\n   🧠 ${reasonText}`)
    
    try {
      const r = await webSearch(q, { maxResults, signal: abortSignal, provider, searxngApiUrl, tavilyApiKey })
      const items = (r.results || []).slice(0, maxResults).map(it => ({ title: String(it.title || ''), url: String(it.url || ''), content: String(it.content || '') }))
      rounds.push({ query: q, items }); 
      
      // 状态更新：搜索返回
      onProgress?.(`📄 搜索成功，获取到 ${items.length} 个页面，正在提取关键信息...`)
    } catch (e: any) {
      const errMsg = e?.message ?? String(e)
      console.error(`[WebSearch][Round ${i + 1}] Search failed for query "${q}":`, errMsg)
      rounds.push({ query: q, items: [], note: errMsg })
      onProgress?.(`❌ 搜索请求失败：${errMsg}`)
      break
    }
  }
  
  if (rounds.length > 0) {
      onProgress?.(`📚 搜索任务结束，正在从 ${rounds.reduce((a,b)=>a+(b.items?.length||0), 0)} 条记录中整理精华...`)
  }
  
  return { rounds, selectedIds }
}

function formatAggregatedSearchForAnswer(rounds: SearchRound[]): string {
  if (!rounds.length) return ''
  let n = 0; const lines: string[] = []; const refLines: string[] = []
  lines.push('【联网搜索结果（已筛选）】')
  rounds.forEach((r, idx) => {
    if (!r.items?.length) return 
    lines.push(`（相关来源：${r.query}）`)
    for (const it of r.items) {
      n++
      lines.push(`[${n}] ${String(it.title || '').trim()}`)
      lines.push(`URL: ${String(it.url || '').trim()}`)
      lines.push(`内容: ${String(it.content || '').trim()}`)
      lines.push('')
      refLines.push(`[${n}] ${String(it.title || '').trim()} - ${String(it.url || '').trim()}`)
    }
  })
  if (n === 0) return '' 

  lines.push('')
  lines.push('【参考来源列表】')
  lines.push(...refLines)
  lines.push('')
  lines.push('【回答要求】')
  lines.push('- 基于以上来源回答用户问题。')
  lines.push('- 引用格式必须为 markdown 链接：[编号](url)，例如 [1](https://example.com)。')
  lines.push('- 可以同时引用多个来源，用逗号分隔：[1](url1), [2](url2)。')
  lines.push('- 在回答末尾，列出所有引用过的参考来源，格式：')
  lines.push('  ## 参考来源')
  lines.push('  - [标题](url)')
  lines.push('- 若来源不足以支持结论，请明确说明。')
  return lines.join('\n')
}

function appendTextToMessageContent(content: MessageContent, appendix: string): MessageContent {
  if (!appendix?.trim()) return content
  if (typeof content === 'string') return `${content}\n\n${appendix}`
  if (Array.isArray(content)) {
    const idx = (content as any[]).findIndex(p => p?.type === 'text' && typeof p?.text === 'string')
    if (idx >= 0) { const arr = [...(content as any[])]; arr[idx] = { ...(arr[idx] || {}), text: `${arr[idx].text}\n\n${appendix}` }; return arr as any }
    return [{ type: 'text', text: appendix }, ...(content as any[])] as any
  }
  return content
}
// ===================== 联网搜索 END =====================

const ErrorCodeMessage: Record<string, string> = { 401: '提供错误的API密钥', 403: '服务器拒绝访问，请稍后再试', 502: '错误的网关', 503: '服务器繁忙，请稍后再试', 504: '网关超时', 500: '服务器繁忙，请稍后再试' }

let auditService: TextAuditService
const _lockedKeys: { key: string; lockedTime: number }[] = []

const DATA_URL_IMAGE_CAPTURE_RE = /data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)/g
const MARKDOWN_IMAGE_RE = /!$$[^$$]*]$\s*([^)]+?)\s*$/g
const UPLOADS_URL_RE = /(\/uploads\/[^)\s>"']+\.(?:png|jpe?g|webp|gif|bmp|heic))/gi
const HTML_IMAGE_RE = /<img[^>]*\ssrc=["']([^"']+)["'][^>]*>/gi
const DATA_URL_IMAGE_RE = /data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g

async function replaceDataUrlImagesWithUploads(text: string): Promise<{ text: string; saved: Array<{ mime: string; filename: string; bytes: number }> }> {
  if (!text) return { text: '', saved: [] }; const saved: Array<{ mime: string; filename: string; bytes: number }> = []; let out = ''; let lastIndex = 0
  const matches = text.matchAll(DATA_URL_IMAGE_CAPTURE_RE)
  for (const m of matches) { const full = m[0]; const mime = m[1]; const base64 = m[2]; const idx = m.index ?? -1; if (idx < 0) continue; out += text.slice(lastIndex, idx); const buffer = Buffer.from(base64, 'base64'); const ext = mime.split('/')[1] || 'png'; const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`; const filePath = path.join(UPLOAD_DIR, filename); await fs.writeFile(filePath, buffer); saved.push({ mime, filename, bytes: buffer.length }); out += `/uploads/${filename}`; lastIndex = idx + full.length }
  out += text.slice(lastIndex); return { text: out, saved }
}

type SavedUpload = { mime: string; filename: string; bytes: number }
type DataUrlCache = Map<string, string>
function parseDataUrlImage(dataUrl: string): { mime: string; base64: string } | null { const m = /^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$/.exec(dataUrl); if (!m) return null; return { mime: m[1] || 'image/png', base64: (m[2] || '').replace(/\s+/g, '') || null } as any }
function mimeToExt(mime: string): string { const t = (mime || '').toLowerCase(); if (t === 'image/jpeg' || t === 'image/jpg') return 'jpg'; if (t === 'image/png') return 'png'; if (t === 'image/webp') return 'webp'; if (t === 'image/gif') return 'gif'; if (t === 'image/bmp') return 'bmp'; if (t === 'image/heic') return 'heic'; return t.split('/')[1] || 'png' }
async function saveDataUrlImageToUploads(dataUrl: string, cache?: DataUrlCache): Promise<{ url: string; saved: SavedUpload } | null> { try { if (!dataUrl?.startsWith('data:image/')) return null; if (cache?.has(dataUrl)) { const url = cache.get(dataUrl)!; return { url, saved: { mime: 'image/*', filename: path.basename(url), bytes: 0 } } }; const parsed = parseDataUrlImage(dataUrl); if (!parsed) return null; await ensureUploadDir(); const buffer = Buffer.from(parsed.base64, 'base64'); const ext = mimeToExt(parsed.mime); const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`; const filePath = path.join(UPLOAD_DIR, filename); await fs.writeFile(filePath, buffer); const url = `/uploads/${filename}`; cache?.set(dataUrl, url); return { url, saved: { mime: parsed.mime, filename, bytes: buffer.length } } } catch (e) { globalThis.console.error('[saveDataUrlImageToUploads] failed:', e); return null } }
async function normalizeMessageContentDataUrlsToUploads(content: MessageContent, cache?: DataUrlCache): Promise<{ content: MessageContent; saved: SavedUpload[] }> { const savedAll: SavedUpload[] = []; if (typeof content === 'string') { const replaced = await replaceDataUrlImagesWithUploads(content); savedAll.push(...replaced.saved); return { content: replaced.text, saved: savedAll } }; if (Array.isArray(content)) { const newParts: any[] = []; for (const p of content as any[]) { if (p?.type === 'text' && typeof p.text === 'string') { const replaced = await replaceDataUrlImagesWithUploads(p.text); savedAll.push(...replaced.saved); newParts.push({ ...p, text: replaced.text }); continue }; if (p?.type === 'image_url' && typeof p.image_url?.url === 'string') { const u = p.image_url.url as string; if (u.startsWith('data:image/')) { const r = await saveDataUrlImageToUploads(u, cache); if (r?.url) { savedAll.push(r.saved); newParts.push({ ...p, image_url: { ...p.image_url, url: r.url } }); continue } }; newParts.push(p); continue }; newParts.push(p) }; return { content: newParts as any, saved: savedAll } }; return { content, saved: savedAll } }
async function normalizeUrlsDataUrlsToUploads(urls: string[], cache?: DataUrlCache): Promise<{ urls: string[]; saved: SavedUpload[] }> { const out: string[] = []; const savedAll: SavedUpload[] = []; for (const u of urls || []) { if (typeof u === 'string' && u.startsWith('data:image/')) { const r = await saveDataUrlImageToUploads(u, cache); if (r?.url) { out.push(r.url); savedAll.push(r.saved) } else { out.push(u) } } else { out.push(u) } }; return { urls: out, saved: savedAll } }
function shortUrlForLog(u: string): string { if (!u) return ''; if (u.startsWith('data:image/')) { return `dataUrl(${u.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,/)?.[1] ?? 'image/*'},len=${u.length})` }; return u.length > 200 ? `${u.slice(0, 200)}...(len=${u.length})` : u }
function extractImageUrlsFromText(text: string): { cleanedText: string; urls: string[] } { if (!text) return { cleanedText: '', urls: [] }; const urls: string[] = []; let cleaned = text; cleaned = cleaned.replace(MARKDOWN_IMAGE_RE, (_m, inside) => { const raw = String(inside ?? '').trim(); if (raw) { const url = raw.split(/\s+/)[0]?.replace(/^<|>$/g, '').trim(); if (url) urls.push(url) }; return '[Image]' }); cleaned = cleaned.replace(HTML_IMAGE_RE, (_m, url) => { if (url) urls.push(url); return '[Image]' }); const rawDataUrls = cleaned.match(DATA_URL_IMAGE_RE); if (rawDataUrls?.length) urls.push(...rawDataUrls); cleaned = cleaned.replace(DATA_URL_IMAGE_RE, '[Image]'); const uploadUrls = cleaned.match(UPLOADS_URL_RE); if (uploadUrls?.length) urls.push(...uploadUrls); cleaned = cleaned.replace(UPLOADS_URL_RE, '[Image]'); return { cleanedText: cleaned, urls } }
async function extractImageUrlsFromMessageContent(content: MessageContent): Promise<{ cleanedText: string; urls: string[] }> { const urls: string[] = []; let cleanedText = ''; if (typeof content === 'string') { const r = extractImageUrlsFromText(content); cleanedText = r.cleanedText; urls.push(...r.urls); if (!cleanedText?.trim() && urls.length) cleanedText = '[Image]'; return { cleanedText, urls } }; if (Array.isArray(content)) { for (const p of content as any[]) { if (p?.type === 'text' && p.text) { const r = extractImageUrlsFromText(p.text); if (r.cleanedText) cleanedText += (cleanedText ? '\n' : '') + r.cleanedText; if (r.urls.length) urls.push(...r.urls) } else if (p?.type === 'image_url' && p.image_url?.url) { urls.push(p.image_url.url); cleanedText += (cleanedText ? '\n' : '') + '[Image]' } }; if (!cleanedText?.trim() && urls.length) cleanedText = '[Image]'; return { cleanedText, urls } }; return { cleanedText: '', urls: [] } }
function dedupeUrlsPreserveOrder(urls: string[]): string[] { const seen = new Set<string>(); const out: string[] = []; for (const u of urls) { if (!u || seen.has(u)) continue; seen.add(u); out.push(u) }; return out }
async function imageUrlToGeminiInlinePart(urlStr: string): Promise<GeminiPart | null> { if (urlStr.startsWith('data:')) { const [header, base64Data] = urlStr.split(','); if (!header || !base64Data) return null; const mimeMatch = header.match(/:(.*?);/); return { inlineData: { mimeType: mimeMatch ? mimeMatch[1] : 'image/png', data: base64Data } } }; if (urlStr.includes('/uploads/')) { const fileData = await getFileBase64(path.basename(urlStr)); if (!fileData) return null; return { inlineData: { mimeType: fileData.mime, data: fileData.data } } }; return null }

type ImageBinding = { tag: string; source: 'current_user' | 'last_ai' | 'prev_user' | 'prev_ai'; url: string }
type HistoryMessageMeta = { role: 'user' | 'model'; cleanedText: string; urls: string[]; messageId: string }
function applyImageBindingsToCleanedText(cleanedText: string, urls: string[], bindings: ImageBinding[]): string { const map = new Map<string, string>(); for (const b of bindings) map.set(b.url, b.tag); let out = cleanedText || ''; for (const u of urls) { const tag = map.get(u); out = out.replace('[Image]', tag ? `[${tag}]` : '[Image]') }; if (!out.trim() && urls.length) out = urls.map(u => (map.get(u) ? `[${map.get(u)}]` : '[Image]')).join(' '); return out }
async function buildGeminiHistoryFromLastMessageId(params: { lastMessageId?: string; maxContextCount: number; forGeminiImageModel: boolean }): Promise<{ metas: HistoryMessageMeta[]; allUrls: string[] }> { const { lastMessageId: startId, maxContextCount, forGeminiImageModel } = params; const metasRev: HistoryMessageMeta[] = []; const urlsRev: string[] = []; let lastMessageId = startId; for (let i = 0; i < maxContextCount; i++) { if (!lastMessageId) break; const msg = await getMessageById(lastMessageId); if (!msg) break; const role = (msg.role === 'assistant' ? 'model' : 'user') as 'user' | 'model'; if (forGeminiImageModel) { const { cleanedText, urls } = await extractImageUrlsFromMessageContent(msg.text); if (urls.length) urlsRev.push(...urls); metasRev.push({ role, cleanedText: cleanedText?.trim() ? cleanedText : (urls.length ? '[Image]' : '[Empty]'), urls, messageId: msg.id }) } else { metasRev.push({ role, cleanedText: typeof msg.text === 'string' ? msg.text : '[Complex Content]', urls: [], messageId: msg.id }) }; lastMessageId = msg.parentMessageId }; return { metas: metasRev.reverse(), allUrls: dedupeUrlsPreserveOrder(urlsRev.reverse()) } }
function getLastNByRole(metas: HistoryMessageMeta[], role: 'user' | 'model', n: number): HistoryMessageMeta[] { const out: HistoryMessageMeta[] = []; for (let i = metas.length - 1; i >= 0; i--) { if (metas[i].role !== role) continue; out.push(metas[i]); if (out.length >= n) break }; return out }
function selectRecentTwoRoundsImages(metas: HistoryMessageMeta[], currentUrls: string[]): ImageBinding[] { const lastModels = getLastNByRole(metas, 'model', 2); const lastUsers = getLastNByRole(metas, 'user', 1); const groups = [{ source: 'current_user' as const, tagPrefix: 'U0', urls: currentUrls }, { source: 'last_ai' as const, tagPrefix: 'A0', urls: lastModels[0]?.urls ?? [] }, { source: 'prev_user' as const, tagPrefix: 'U1', urls: lastUsers[0]?.urls ?? [] }, { source: 'prev_ai' as const, tagPrefix: 'A1', urls: lastModels[1]?.urls ?? [] }]; const seen = new Set<string>(); const bindings: ImageBinding[] = []; for (const g of groups) { let idx = 0; for (const u of g.urls) { if (!u || seen.has(u)) continue; seen.add(u); idx += 1; bindings.push({ tag: `${g.tagPrefix}_${idx}`, source: g.source, url: u }) } }; return bindings }
const IMAGE_SOURCE_LABEL: Record<ImageBinding['source'], string> = { current_user: '本次用户消息', last_ai: '最近一条AI回复', prev_user: '上一次用户消息', prev_ai: '倒数第二条AI回复' }
export async function initApi(key: KeyConfig, { model, maxContextCount, temperature, top_p, abortSignal, content, systemMessage, lastMessageId, isImageModel }: any) {
  const config = await getCacheConfig()
  const OPENAI_API_BASE_URL = isNotEmptyString(key.baseUrl) ? key.baseUrl : config.apiBaseUrl
  const openai = new OpenAI({ baseURL: OPENAI_API_BASE_URL, apiKey: key.key, maxRetries: 0, timeout: config.timeoutMs })
  const modelConfig = MODEL_CONFIGS[model] || { supportTopP: true }
  const finalTemperature = modelConfig.defaultTemperature ?? temperature
  const shouldUseTopP = modelConfig.supportTopP
  const messages: OpenAI.ChatCompletionMessageParam[] = []
  for (let i = 0; i < maxContextCount; i++) { if (!lastMessageId) break; const message = await getMessageById(lastMessageId); if (!message) break; let safeContent = message.text as string; if (typeof safeContent === 'string') safeContent = safeContent.replace(DATA_URL_IMAGE_RE, '[Image Data Removed]'); messages.push({ role: message.role as any, content: safeContent }); lastMessageId = message.parentMessageId }
  if (systemMessage) messages.push({ role: 'system', content: systemMessage })
  messages.reverse(); messages.push({ role: 'user', content })
  const enableStream = !isImageModel
  const options: OpenAI.ChatCompletionCreateParams = { model, stream: enableStream, stream_options: enableStream ? { include_usage: true } : undefined, messages }
  options.temperature = finalTemperature; if (shouldUseTopP) options.top_p = top_p
  try { const siteCfg = config.siteConfig; const reasoningModelsStr = siteCfg?.reasoningModels || ''; const reasoningEffort = siteCfg?.reasoningEffort || 'medium'; const reasoningModelList = reasoningModelsStr.split(/[,，]/).map(s => s.trim()).filter(Boolean); if (reasoningModelList.includes(model) && reasoningEffort && reasoningEffort !== 'none') (options as any).reasoning_effort = reasoningEffort } catch (e) { globalThis.console.error('[OpenAI] set reasoning_effort failed:', e) }
  if (API_DEBUG) { debugLog('====== [OpenAI Request Debug] ======'); debugLog('[baseURL]', OPENAI_API_BASE_URL); debugLog('[model]', model); debugLog('[messagesSummary]', safeJson(summarizeOpenAIMessages(messages))); debugLog('====== [OpenAI Request Debug End] ======') }
  return await openai.chat.completions.create(options, { signal: abortSignal })
}

type ProcessThread = { key: string; userId: string; roomId: number; abort: AbortController; messageId: string }
const processThreads: ProcessThread[] = []
const makeThreadKey = (userId: string, roomId: number) => `${userId}:${roomId}`

async function chatReplyProcess(options: RequestOptions): Promise<{ message: string; data: ChatResponse; status: string }> {
  const userId = options.user._id.toString(); const messageId = options.messageId; const roomId = options.room.roomId; const abort = new AbortController(); const customMessageId = generateMessageId(); const threadKey = makeThreadKey(userId, roomId)
  const existingIdx = processThreads.findIndex(t => t.key === threadKey); if (existingIdx > -1) { processThreads[existingIdx].abort.abort(); processThreads.splice(existingIdx, 1) }
  processThreads.push({ key: threadKey, userId, abort, messageId, roomId })
  await ensureUploadDir(); const model = options.room.chatModel; const key = await getRandomApiKey(options.user, model, options.room.accountId); const maxContextCount = options.user.advanced.maxContextCount ?? 20
  if (!key) { const idx = processThreads.findIndex(d => d.key === threadKey); if (idx > -1) processThreads.splice(idx, 1); throw new Error('没有对应的 apikeys 配置，请再试一次。') }
  updateRoomChatModel(userId, options.room.roomId, model)
  const { message, uploadFileKeys, lastContext, process: processCb, systemMessage, temperature, top_p } = options
  processCb?.({ id: customMessageId, text: '', role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })

  let content: MessageContent = message; let fileContext = ''
  const globalConfig = await getCacheConfig(); const imageModelsStr = globalConfig.siteConfig.imageModels || ''; const imageModelList = imageModelsStr.split(/[,，]/).map(s => s.trim()).filter(Boolean)
  const isImage = imageModelList.some(m => model.includes(m)); const isGeminiImageModel = isImage && model.includes('gemini')

  if (uploadFileKeys && uploadFileKeys.length > 0) {
    const textFiles = uploadFileKeys.filter(k => isTextFile(k)); const imageFiles = uploadFileKeys.filter(k => isImageFile(k))
    if (textFiles.length > 0) { for (const fileKey of textFiles) { try { const filePath = path.join(UPLOAD_DIR, stripTypePrefix(fileKey)); await fs.access(filePath); const fileContent = await fs.readFile(filePath, 'utf-8'); fileContext += `\n\n--- File Start: ${stripTypePrefix(fileKey)} ---\n${fileContent}\n--- File End ---\n` } catch (e) { globalThis.console.error(`Error reading text file ${fileKey}`, e); fileContext += `\n\n[System Error: File ${fileKey} not found or unreadable]\n` } } }
    const finalMessage = message + (fileContext ? `\n\nAttached Files content:\n${fileContext}` : '')
    if (imageFiles.length > 0) { content = [{ type: 'text', text: finalMessage }]; for (const uploadFileKey of imageFiles) { content.push({ type: 'image_url', image_url: { url: await convertImageUrl(stripTypePrefix(uploadFileKey)) } }) } }
    else { content = finalMessage }
  }

  // 初始化一个变量来存储搜索过程的日志，确保在生成最终答案时，用户能看到之前发生了什么
  let searchProcessLog = ''

  // ===== 联网搜索 =====
  const allowSearch = globalConfig.siteConfig?.webSearchEnabled === true
  const finalSearchMode = allowSearch && options.searchMode === true && !isImage

  if (finalSearchMode) {
    try {
      const plannerModelCfg = String(globalConfig.siteConfig?.webSearchPlannerModel ?? '').trim()
      const plannerModelEnv = String(process.env.WEB_SEARCH_PLANNER_MODEL ?? '').trim()
      const plannerModelName = plannerModelCfg || plannerModelEnv || model

      // ✅ 为 planner 找独立 key；找不到则 fallback 到当前对话模型+key
      let plannerKey = key
      let actualPlannerModel = plannerModelName
      if (plannerModelName !== model) {
        const candidateKey = await getRandomApiKey(options.user, plannerModelName)
        if (candidateKey) {
          plannerKey = candidateKey
        }
        else {
          console.warn(`[WebSearch] No key found for planner model "${plannerModelName}", falling back to current model "${model}"`)
          actualPlannerModel = model
          plannerKey = key
        }
      }

      const PLANNER_BASE_URL = isNotEmptyString(plannerKey.baseUrl) ? plannerKey.baseUrl : globalConfig.apiBaseUrl
      const plannerOpenai = new OpenAI({ baseURL: PLANNER_BASE_URL, apiKey: plannerKey.key, maxRetries: 0, timeout: globalConfig.timeoutMs })

      const maxRounds = Math.max(1, Math.min(6, Number(globalConfig.siteConfig?.webSearchMaxRounds ?? process.env.WEB_SEARCH_MAX_ROUNDS ?? 3)))
      const maxResults = Math.max(1, Math.min(10, Number(globalConfig.siteConfig?.webSearchMaxResults ?? process.env.WEB_SEARCH_MAX_RESULTS ?? 5)))

      // ✅ diff plannerModels
      const plannerModels = [actualPlannerModel, model].filter((v, i, arr) => Boolean(v) && arr.indexOf(v) === i)

      const searchProvider = globalConfig.siteConfig?.webSearchProvider as any
      const searchSearxngUrl = String(globalConfig.siteConfig?.searxngApiUrl ?? '').trim() || undefined
      const searchTavilyKey = String(globalConfig.siteConfig?.tavilyApiKey ?? '').trim() || undefined

      const progressMessages: string[] = []
      const onProgressLocal = (status: string) => {
        progressMessages.push(status)
        const displayLog = progressMessages.join('\n')
        // 实时推送到前端，让用户看到“思考中...”
        processCb?.({
          id: customMessageId, text: displayLog + '\n\n⏳ 正在执行...', role: 'assistant',
          conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined,
        })
      }

      // ✅ Build Context and Date
      const historyContext = await buildConversationContext(lastContext?.parentMessageId, maxContextCount)
      const currentDate = new Date().toLocaleString()

      const { rounds, selectedIds } = await runIterativeWebSearch({
        openai: plannerOpenai, plannerModels, userQuestion: message, maxRounds, maxResults,
        abortSignal: abort.signal, provider: searchProvider, searxngApiUrl: searchSearxngUrl, tavilyApiKey: searchTavilyKey,
        onProgress: onProgressLocal,
        fullContext: historyContext, date: currentDate
      })

      // 保存日志供最终输出使用
      if (progressMessages.length > 0) {
        searchProcessLog = progressMessages.join('\n') + '\n\n---\n\n'
      }

      // ✅ Filter results based on selectedIds, NO quantity limits
      const filteredRounds = rounds.map((r, rIdx) => ({
        query: r.query,
        note: r.note,
        items: r.items.filter((_, iIdx) => selectedIds.has(`${rIdx + 1}.${iIdx + 1}`))
      })).filter(r => r.items.length > 0 || r.note)
      
      const ctx = formatAggregatedSearchForAnswer(filteredRounds)
      
      // 核心修改：明确告知用户正在生成最终回答
      let finalStatusMessage = '✅ 资料整理完毕，正在生成回答...' 
      if (!ctx && rounds.length > 0) finalStatusMessage = '⚠️ 未能筛选出有效引用，尝试直接回答...'

      processCb?.({
        id: customMessageId, 
        text: searchProcessLog + `⚡️ ${finalStatusMessage}\n(模型正在阅读 ${ctx.length > 5000 ? '大量' : ''}资料并构思最终回答，请稍候...)`, 
        role: 'assistant',
        conversationId: lastContext?.conversationId, 
        parentMessageId: lastContext?.parentMessageId, 
        detail: undefined,
      })

      if (ctx) content = appendTextToMessageContent(content, ctx)

      if (API_DEBUG) {
        debugLog('====== [WebSearch Debug] ======')
        debugLog('[configuredPlanner]', plannerModelName, '[actualPlanner]', actualPlannerModel)
        debugLog('[rounds]', rounds.length, '[filteredItems]', filteredRounds.reduce((s, r) => s + (r.items?.length ?? 0), 0))
        debugLog('====== [WebSearch Debug End] ======')
      }
    }
    catch (e: any) { 
       if (isAbortError(e, abort.signal)) throw e; 
       globalThis.console.error('[WebSearch] failed:', e?.message ?? e);
       searchProcessLog += `\n❌ 联网搜索模块遇到问题：${e?.message ?? '未知错误'}\n\n---\n\n`
       processCb?.({
          id: customMessageId, text: searchProcessLog + '尝试使用已有知识回答...', role: 'assistant',
          conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined
       })
    }
  }

  try {
    // ① Gemini
    if (isGeminiImageModel) {
      const baseUrl = isNotEmptyString(key.baseUrl) ? key.baseUrl.replace(/\/+$/, '') : undefined; const dataUrlCache: DataUrlCache = new Map()
      const normalizedCurrent = await normalizeMessageContentDataUrlsToUploads(content, dataUrlCache); content = normalizedCurrent.content
      const { metas } = await buildGeminiHistoryFromLastMessageId({ lastMessageId: lastContext.parentMessageId, maxContextCount, forGeminiImageModel: true })
      const metasNormalized: HistoryMessageMeta[] = []; for (const m of metas) { const nu = await normalizeUrlsDataUrlsToUploads(m.urls || [], dataUrlCache); metasNormalized.push({ ...m, urls: nu.urls }) }
      const { cleanedText: currentCleanedText, urls: currentUrlsRaw } = await extractImageUrlsFromMessageContent(content); const currentUrlsNorm = await normalizeUrlsDataUrlsToUploads(currentUrlsRaw || [], dataUrlCache); const currentUrls = currentUrlsNorm.urls
      const bindings = selectRecentTwoRoundsImages(metasNormalized, currentUrls)
      const history = metasNormalized.map(m => ({ role: m.role, parts: [{ text: applyImageBindingsToCleanedText(m.cleanedText, m.urls, bindings) || '[Empty]' }] }))
      const userInstructionBase = (currentCleanedText && currentCleanedText.trim()) ? currentCleanedText.trim() : '请继续生成/修改图片。'
      const labeledUserInstruction = applyImageBindingsToCleanedText(userInstructionBase, currentUrls, bindings)
      const inputParts: GeminiPart[] = []
      if (bindings.length) { inputParts.push({ text: `【最近两轮图片映射表】\n${bindings.map(b => `${b.tag}=${IMAGE_SOURCE_LABEL[b.source]}(${shortUrlForLog(b.url)})`).join('\n')}\n` }) }
      for (const b of bindings) { inputParts.push({ text: `【${b.tag}｜${IMAGE_SOURCE_LABEL[b.source]}】` }); const imgPart = await imageUrlToGeminiInlinePart(b.url); if (imgPart) inputParts.push(imgPart); else inputParts.push({ text: `（该图片无法以内联方式上传：${shortUrlForLog(b.url)}）` }) }
      inputParts.push({ text: `【编辑指令】${labeledUserInstruction}` })
      const ai = new GoogleGenAI({ apiKey: key.key, ...(baseUrl ? { httpOptions: { baseUrl } } : {}) })
      const response = await abortablePromise(ai.models.generateContent({ model, contents: [...history, { role: 'user', parts: inputParts }], config: { responseModalities: ['TEXT', 'IMAGE'], imageConfig: { aspectRatio: '16:9', imageSize: '4K' }, ...(systemMessage ? { systemInstruction: systemMessage } as any : {}) } as any } as any), abort.signal)
      if (!response.candidates || response.candidates.length === 0) throw new Error('[Gemini] Empty candidates.')
      let text = searchProcessLog; // 保留搜索日志
      const parts = response.candidates?.[0]?.content?.parts ?? []
      for (const part of parts as any[]) { if (part?.text) { const replaced = await replaceDataUrlImagesWithUploads(part.text as string); text += replaced.text }; const inline = part?.inlineData; if (inline?.data) { const mime = inline.mimeType || 'image/png'; const buffer = Buffer.from(inline.data as string, 'base64'); const ext = mime.split('/')[1] || 'png'; const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`; await fs.writeFile(path.join(UPLOAD_DIR, filename), buffer); text += `${text ? '\n\n' : ''}![Generated Image](/uploads/${filename})` } }
      if (!text) text = '[Gemini] Success but no text/image parts returned.'
      processCb?.({ id: customMessageId, text, role: 'assistant', conversationId: lastContext.conversationId, parentMessageId: lastContext.parentMessageId, detail: undefined })
      return sendResponse({ type: 'Success', data: { object: 'chat.completion', choices: [{ message: { role: 'assistant', content: text }, finish_reason: 'stop', index: 0, logprobs: null }], created: Date.now(), conversationId: lastContext.conversationId, model, text, id: customMessageId, detail: {} } })
    }

    // ② OpenAI
    const api = await initApi(key, { model, maxContextCount, temperature, top_p, content, abortSignal: abort.signal, systemMessage, lastMessageId: lastContext.parentMessageId, isImageModel: isImage })
    let text = searchProcessLog; // 初始化文本包含搜索日志
    let chatIdRes = customMessageId; let modelRes = ''; let usageRes: any
    if (isImage) {
      const response = api as any; const choice = response.choices[0]; let rawContent = choice.message?.content || ''; modelRes = response.model; usageRes = response.usage
      if (rawContent && !rawContent.startsWith('![') && (rawContent.startsWith('http') || rawContent.startsWith('data:image'))) text += `![Generated Image](${rawContent})`; else text += rawContent
      processCb?.({ id: customMessageId, text, role: choice.message.role || 'assistant', conversationId: lastContext.conversationId, parentMessageId: lastContext.parentMessageId, detail: { choices: [{ finish_reason: 'stop', index: 0, logprobs: null, message: choice.message }], created: response.created, id: response.id, model: response.model, object: 'chat.completion', usage: response.usage } as any })
    } else {
      // Stream 处理：chunks 逐渐追加到 text (已经包含 searchProcessLog)
      for await (const chunk of api as AsyncIterable<OpenAI.ChatCompletionChunk>) { 
          text += chunk.choices[0]?.delta.content ?? ''; 
          chatIdRes = customMessageId; modelRes = chunk.model; usageRes = usageRes || chunk.usage; 
          processCb?.({ ...chunk, id: customMessageId, text, role: chunk.choices[0]?.delta.role || 'assistant', conversationId: lastContext.conversationId, parentMessageId: lastContext.parentMessageId }) 
      }
    }
    return sendResponse({ type: 'Success', data: { object: 'chat.completion', choices: [{ message: { role: 'assistant', content: text }, finish_reason: 'stop', index: 0, logprobs: null }], created: Date.now(), conversationId: lastContext.conversationId, model: modelRes, text, id: chatIdRes, detail: { usage: usageRes && { ...usageRes, estimated: false } } } })
  } catch (error: any) {
    if (isAbortError(error, abort.signal)) throw error
    const code = error.statusCode
    if (code === 429 && (error.message.includes('Too Many Requests') || error.message.includes('Rate limit'))) { if (options.tryCount++ < 3) { _lockedKeys.push({ key: key.key, lockedTime: Date.now() }); await new Promise(resolve => setTimeout(resolve, 2000)); const index = processThreads.findIndex(d => d.key === threadKey); if (index > -1) processThreads.splice(index, 1); return await chatReplyProcess(options) } }
    globalThis.console.error(error)
    if (Reflect.has(ErrorCodeMessage, code)) return sendResponse({ type: 'Fail', message: ErrorCodeMessage[code] })
    return sendResponse({ type: 'Fail', message: error.message ?? 'Please check the back-end console' })
  } finally { const index = processThreads.findIndex(d => d.key === threadKey); if (index > -1) processThreads.splice(index, 1) }
}

export function abortChatProcess(userId: string, roomId: number) {
  const key = makeThreadKey(userId, roomId); const index = processThreads.findIndex(d => d.key === key); if (index <= -1) return
  const messageId = processThreads[index].messageId; processThreads[index].abort.abort(); processThreads.splice(index, 1)
  if (API_DEBUG) { debugLog('====== [Abort Debug] ======'); debugLog('[userId]', userId, '[roomId]', roomId, '[messageId]', messageId); debugLog('====== [Abort Debug End] ======') }
  return messageId
}

export function initAuditService(audit: AuditConfig) { if (!audit || !audit.options || !audit.options.apiKey || !audit.options.apiSecret) return; const Service = textAuditServices[audit.provider]; auditService = new Service(audit.options) }

async function containsSensitiveWords(audit: AuditConfig, text: string): Promise<boolean> {
  if (audit.customizeEnabled && isNotEmptyString(audit.sensitiveWords)) { const textLower = text.toLowerCase(); if (audit.sensitiveWords.split('\n').filter(d => textLower.includes(d.trim().toLowerCase())).length > 0) return true }
  if (audit.enabled) { if (!auditService) initAuditService(audit); return await auditService.containsSensitiveWords(text) }
  return false
}

async function chatConfig() { const config = await getOriginConfig() as ModelConfig; return sendResponse<ModelConfig>({ type: 'Success', data: config }) }
async function getMessageById(id: string): Promise<ChatMessage | undefined> {
  const isPrompt = id.startsWith('prompt_')
  const chatInfo = await getChatByMessageId(isPrompt ? id.substring(7) : id)
  if (!chatInfo) return undefined
  const parentMessageId = isPrompt ? chatInfo.options.parentMessageId : `prompt_${id}`
  if (chatInfo.status !== Status.Normal) return parentMessageId ? getMessageById(parentMessageId) : undefined

  if (isPrompt) {
    let promptText = chatInfo.prompt
    const allFileKeys = chatInfo.images || []; const textFiles = allFileKeys.filter(k => isTextFile(k)); const imageFiles = allFileKeys.filter(k => isImageFile(k))
    if (textFiles.length > 0) { let fileContext = ''; for (const fileKey of textFiles) { try { const filePath = path.join(UPLOAD_DIR, stripTypePrefix(fileKey)); const fileContent = await fs.readFile(filePath, 'utf-8'); fileContext += `\n\n--- Context File: ${stripTypePrefix(fileKey)} ---\n${fileContent}\n--- End File ---\n` } catch { } }; promptText += (fileContext ? `\n\n[Attached Files History]:\n${fileContext}` : '') }
    if (promptText && typeof promptText === 'string') promptText = promptText.replace(DATA_URL_IMAGE_RE, '[Image Data Removed]')
    let content: MessageContent = promptText
    if (imageFiles.length > 0) { content = [{ type: 'text', text: promptText }]; for (const image of imageFiles) { content.push({ type: 'image_url', image_url: { url: await convertImageUrl(stripTypePrefix(image)) } }) } }
    if (API_DEBUG) { debugLog('------ [getMessageById Prompt Debug] ------'); debugLog('[id]', id, '[parent]', parentMessageId); debugLog('[promptTextLen]', typeof promptText === 'string' ? promptText.length : -1); debugLog('[promptTextPreview]', trunc(promptText)); debugLog('[imageFiles]', imageFiles); debugLog('[textFiles]', textFiles); debugLog('------------------------------------------') }
    return { id, conversationId: chatInfo.options.conversationId, parentMessageId, role: 'user', text: content }
  } else {
    let responseText = chatInfo.response || ''
    if (responseText && typeof responseText === 'string') responseText = responseText.replace(DATA_URL_IMAGE_RE, '[Image History]')
    if (API_DEBUG) { debugLog('------ [getMessageById Assistant Debug] ------'); debugLog('[id]', id, '[parent]', parentMessageId); debugLog('[responseLen]', typeof responseText === 'string' ? responseText.length : -1); debugLog('[responsePreview]', trunc(responseText)); debugLog('---------------------------------------------') }
    return { id, conversationId: chatInfo.options.conversationId, parentMessageId, role: 'assistant', text: responseText }
  }
}

async function randomKeyConfig(keys: KeyConfig[]): Promise<KeyConfig | null> {
  if (keys.length <= 0) return null
  _lockedKeys.filter(d => d.lockedTime <= Date.now() - 1000 * 20).forEach(d => _lockedKeys.splice(_lockedKeys.indexOf(d), 1))
  let unsedKeys = keys.filter(d => _lockedKeys.filter(l => d.key === l.key).length <= 0); const start = Date.now()
  while (unsedKeys.length <= 0) { if (Date.now() - start > 3000) break; await new Promise(resolve => setTimeout(resolve, 1000)); unsedKeys = keys.filter(d => _lockedKeys.filter(l => d.key === l.key).length <= 0) }
  if (unsedKeys.length <= 0) return null; return unsedKeys[Math.floor(Math.random() * unsedKeys.length)]
}

async function getRandomApiKey(user: UserInfo, chatModel: string, accountId?: string): Promise<KeyConfig | undefined> {
  let keys = (await getCacheApiKeys()).filter(d => hasAnyRole(d.userRoles, user.roles)).filter(d => d.chatModels.includes(chatModel))
  if (accountId) keys = keys.filter(d => d.keyModel === 'ChatGPTUnofficialProxyAPI' && getAccountId(d.key) === accountId)
  const picked = await randomKeyConfig(keys)
  if (API_DEBUG) { debugLog('====== [Key Pick Debug] ======'); debugLog('[chatModel]', chatModel, '[accountId]', accountId ?? '(none)'); debugLog('[candidateKeyCount]', keys.length); debugLog('[picked]', picked ? { keyModel: picked.keyModel, baseUrl: picked.baseUrl, remark: picked.remark } : '(null)'); debugLog('====== [Key Pick Debug End] ======') }
  return picked ?? undefined
}

function getAccountId(accessToken: string): string { try { const jwt = jwt_decode(accessToken) as JWT; return jwt['https://api.openai.com/auth'].user_id } catch { return '' } }

export { chatReplyProcess, chatConfig, containsSensitiveWords }
