import * as dotenv from 'dotenv'
import { generateText, streamText } from 'ai'
import type { CoreMessage } from 'ai'
import { textTokens } from 'gpt-token'
import type { TiktokenModel } from 'gpt-token'
import type { AuditConfig, KeyConfig, UserInfo } from '../storage/model'
import { Status } from '../storage/model'
import { convertImageUrl } from '../utils/image'
import type { TextAuditService } from '../utils/textAudit'
import { textAuditServices } from '../utils/textAudit'
import { getCacheApiKeys, getCacheConfig, getOriginConfig } from '../storage/config'
import { sendResponse } from '../utils'
import { hasAnyRole, isNotEmptyString } from '../utils/is'
import type { ModelConfig } from '../types'
import { getChatByMessageId, updateRoomChatModel } from '../storage/mongo'
import type { ChatMessage, ChatResponse, MessageContent, RequestOptions } from './types'
import * as fs from 'node:fs/promises'
import * as path from 'node:path'
import { generateMessageId } from '../utils/id-generator'
import { isAbortError } from './abortable'
import { webSearch } from '../utils/webSearch'
import { createLanguageModel, normalizeProvider } from '../ai/provider'
import type { KeyProvider } from '../ai/provider'

dotenv.config()

const API_DEBUG = process.env.API_DEBUG === 'true'
const API_DEBUG_MAX_TEXT = Number(process.env.API_DEBUG_MAX_TEXT ?? 800)

function trunc(s: any, n = API_DEBUG_MAX_TEXT): string {
  const str = typeof s === 'string' ? s : JSON.stringify(s ?? '')
  if (!str) return ''
  return str.length > n ? `${str.slice(0, n)}...(truncated,total=${str.length})` : str
}
function safeJson(obj: any): string { try { return JSON.stringify(obj, null, 2) } catch (e: any) { return `[Unserializable: ${e?.message ?? e}]` } }
function debugLog(...args: any[]) { if (!API_DEBUG) return; console.log(...args) }

/**
 * ✅ 判断是否为 OpenAI 兼容类型（包含 completions 和 responses）
 */
function isOpenAILike(provider: string): boolean {
  return provider === 'openai-completions' || provider === 'openai-responses'
}

const MODEL_CONFIGS: Record<string, { supportTopP: boolean; defaultTemperature?: number }> = {
  'gpt-5-search-api': { supportTopP: false, defaultTemperature: 0.8 },
}

const UPLOAD_DIR = path.resolve(process.cwd(), 'uploads')

async function ensureUploadDir() {
  try { await fs.access(UPLOAD_DIR) } catch { await fs.mkdir(UPLOAD_DIR, { recursive: true }) }
}

function stripTypePrefix(key: string): string { return key.replace(/^(img:|txt:|file:)/, '') }

function isTextFile(filename: string): boolean {
  if (filename.startsWith('txt:')) return true
  if (filename.startsWith('img:') || filename.startsWith('file:')) return false
  const ext = path.extname(stripTypePrefix(filename)).toLowerCase()
  return ['.txt', '.md', '.json', '.csv', '.js', '.ts', '.py', '.java', '.html', '.css', '.xml', '.yml', '.yaml', '.log', '.ini', '.config'].includes(ext)
}

function isImageFile(filename: string): boolean {
  if (filename.startsWith('img:')) return true
  if (filename.startsWith('txt:') || filename.startsWith('file:')) return false
  const ext = path.extname(stripTypePrefix(filename)).toLowerCase()
  return ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.heic', '.bmp'].includes(ext)
}

function isMultiModalFile(filename: string): boolean {
  if (filename.startsWith('file:')) return true
  if (filename.startsWith('img:') || filename.startsWith('txt:')) return false
  const ext = path.extname(stripTypePrefix(filename)).toLowerCase()
  return ['.pdf', '.mp4', '.mpeg', '.mov', '.mpg', '.avi', '.wmv', '.flv', '.webm', '.mpegps', '.mp3', '.wav', '.ogg'].includes(ext)
}

async function getFileBase64(filename: string): Promise<{ mime: string; data: string } | null> {
  try {
    const realFilename = stripTypePrefix(filename)
    const filePath = path.join(UPLOAD_DIR, realFilename)
    await fs.access(filePath)
    const buffer = await fs.readFile(filePath)
    const ext = path.extname(realFilename).toLowerCase().replace('.', '')

    let mime = 'application/octet-stream'
    if (ext === 'jpg' || ext === 'jpeg') mime = 'image/jpeg'
    else if (ext === 'png') mime = 'image/png'
    else if (ext === 'webp') mime = 'image/webp'
    else if (ext === 'gif') mime = 'image/gif'
    else if (ext === 'heic') mime = 'image/heic'
    else if (ext === 'bmp') mime = 'image/bmp'
    else if (ext === 'pdf') mime = 'application/pdf'
    else if (ext === 'mp4') mime = 'video/mp4'
    else if (ext === 'mov') mime = 'video/quicktime'
    else if (ext === 'avi') mime = 'video/x-msvideo'
    else if (ext === 'wmv') mime = 'video/x-ms-wmv'
    else if (ext === 'flv') mime = 'video/x-flv'
    else if (ext === 'webm') mime = 'video/webm'
    else if (['mpeg', 'mpg', 'mpegps'].includes(ext)) mime = 'video/mpeg'
    else if (ext === 'mp3') mime = 'audio/mp3'
    else if (ext === 'wav') mime = 'audio/wav'
    else if (ext === 'ogg') mime = 'audio/ogg'

    return { mime, data: buffer.toString('base64') }
  }
  catch (e) {
    globalThis.console.error(`[File Read Error] ${filename}:`, e)
    return null
  }
}

function estimateTokenCount(messages: Array<{ role: string; content: any }>): number {
  let allText = ''
  try {
    allText = (messages ?? []).map(m => {
      const c = m?.content
      if (typeof c === 'string') return c
      if (Array.isArray(c)) { return c.map((p: any) => p?.type === 'text' ? String(p.text || '') : '').join('') }
      return ''
    }).join('\n')
    const count = textTokens(allText, 'gpt-3.5-turbo' as TiktokenModel)
    if (count === 0 && allText.length > 0) return Math.ceil(allText.length / 3)
    return count
  }
  catch {
    if (allText && allText.length > 0) return Math.ceil(allText.length / 4)
    return 0
  }
}

function mergeAbortSignals(signals: Array<AbortSignal | undefined>): AbortSignal | undefined {
  const list = signals.filter(Boolean) as AbortSignal[]
  if (!list.length) return undefined
  const anyFn = (AbortSignal as any).any
  if (typeof anyFn === 'function') return anyFn(list)
  const ac = new AbortController()
  const onAbort = () => { try { ac.abort() } catch { } }
  for (const s of list) { if (s.aborted) { try { ac.abort() } catch { } break }; s.addEventListener('abort', onAbort, { once: true }) }
  return ac.signal
}

function toCoreMessages(messages: Array<{ role: string; content: any }>): CoreMessage[] {
  return (messages || []).map((m) => {
    const role = (m.role === 'system' || m.role === 'user' || m.role === 'assistant') ? m.role : 'user'
    const c = m.content
    if (typeof c === 'string') { return { role, content: c } }
    if (Array.isArray(c)) {
      const parts = c.map((p: any) => {
        if (p?.type === 'text') return { type: 'text' as const, text: String(p.text ?? '') }
        if (p?.type === 'image_url') return { type: 'image' as const, image: String(p.image_url?.url ?? '') }
        if (p?.type === 'file') return { type: 'file' as const, data: p.data, mimeType: p.mimeType }
        return { type: 'text' as const, text: String(p?.text ?? '') }
      })
      return { role, content: parts as any }
    }
    return { role, content: String(c ?? '') }
  })
}

function toUsageResponse(usage: any): any | undefined {
  if (!usage) return undefined
  const input = Number(usage.inputTokens ?? usage.promptTokens ?? usage.prompt_tokens ?? 0)
  const output = Number(usage.outputTokens ?? usage.completionTokens ?? usage.completion_tokens ?? 0)
  const total = Number(usage.totalTokens ?? usage.total_tokens ?? (input + output))
  return { prompt_tokens: input, completion_tokens: output, total_tokens: total, estimated: false }
}

const DATA_URL_IMAGE_RE = /data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g

async function loadContextMessages(params: {
  maxContextCount: number
  lastMessageId?: string
  systemMessage?: string
  content: MessageContent
}): Promise<Array<{ role: 'system' | 'user' | 'assistant'; content: any }>> {
  const { maxContextCount, systemMessage, content } = params
  let { lastMessageId } = params

  const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: any }> = []
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId) break
    const message = await getMessageById(lastMessageId)
    if (!message) break
    let safeContent = message.text as any
    if (typeof safeContent === 'string') safeContent = safeContent.replace(DATA_URL_IMAGE_RE, '[Image Data Removed]')
    messages.push({ role: message.role, content: safeContent })
    lastMessageId = message.parentMessageId
  }

  if (systemMessage) messages.push({ role: 'system', content: systemMessage })
  messages.reverse()
  messages.push({ role: 'user', content: content as any })
  return messages
}

type SearchPlan = { action: 'search' | 'stop'; query?: string; reason?: string; selected_ids?: string[]; context_summary?: string }
type SearchRound = { query: string; items: Array<{ title: string; url: string; content: string }>; note?: string }

function safeParseJsonFromText(text: string): any | null {
  if (!text) return null
  const s = String(text).trim()
  try { return JSON.parse(s) } catch { }
  const i = s.indexOf('{'); const j = s.lastIndexOf('}')
  if (i >= 0 && j > i) { try { return JSON.parse(s.slice(i, j + 1)) } catch { } }
  return null
}

function removeImagesFromText(text: string): string {
  if (!text) return ''
  let clean = text.replace(/!$$[^$$]*\]\s*$[^)]+?$\s*$/g, '')
  clean = clean.replace(/<img[^>]*>/gi, '')
  clean = clean.replace(/data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g, '')
  return clean.trim()
}

class PlannerTimeoutError extends Error {
  public readonly code = 'PLANNER_TIMEOUT'
  public readonly kind: string
  public readonly original?: any
  constructor(public readonly timeoutMs: number, opts?: { kind?: string; original?: any }) {
    const kind = opts?.kind || 'timeout'
    const original = opts?.original
    const origMsg = original?.message ? String(original.message) : (original ? String(original) : '')
    super(`Planner timeout (${kind}) after ${timeoutMs}ms${origMsg ? `: ${origMsg}` : ''}`)
    this.name = 'PlannerTimeoutError'
    this.kind = kind
    this.original = original
    try { (this as any).cause = original } catch { }
  }
}

function sleep(ms: number) { return new Promise(resolve => setTimeout(resolve, ms)) }

function oneLine(s: any, maxLen = 260): string {
  const str = String(s ?? '').replace(/\s+/g, ' ').trim()
  if (!str) return ''
  return str.length > maxLen ? `${str.slice(0, maxLen)}...(len=${str.length})` : str
}

function getStatusCodeLike(e: any): number | undefined {
  const raw = e?.statusCode ?? e?.status ?? e?.response?.status ?? e?.error?.status ?? e?.error?.statusCode
  const n = typeof raw === 'number' ? raw : Number(raw)
  return Number.isFinite(n) ? n : undefined
}

function getErrCodeLike(e: any): string | undefined {
  const c = e?.code ?? e?.error?.code ?? e?.cause?.code
  return c ? String(c) : undefined
}

function summarizeAnyError(e: any): string {
  if (!e) return 'unknown'
  const status = getStatusCodeLike(e)
  const name = e?.name ? String(e.name) : undefined
  const code = getErrCodeLike(e)
  const msg = oneLine(e?.message ?? e)
  const parts: string[] = []
  if (status) parts.push(`status=${status}`)
  if (name) parts.push(`name=${name}`)
  if (code) parts.push(`code=${code}`)
  if (msg) parts.push(`msg=${msg}`)
  return parts.join(', ') || oneLine(e)
}

function summarizePlannerTimeoutReason(e: any): string {
  if (e instanceof PlannerTimeoutError) {
    const orig = e.original ? summarizeAnyError(e.original) : ''
    return [`kind=${e.kind}`, `timeout=${Math.round(e.timeoutMs / 1000)}s`, orig ? `original={${orig}}` : '']
      .filter(Boolean)
      .join(', ')
  }
  return summarizeAnyError(e)
}

function isLikelySdkOrNetworkTimeout(e: any): boolean {
  const status = getStatusCodeLike(e)
  if (status && (status === 504 || status === 503 || status === 502 || status === 500)) return true
  const name = String(e?.name ?? '')
  if (name === 'TimeoutError' || name === 'AbortError') return true
  const code = String(getErrCodeLike(e) ?? '')
  if (code === 'ETIMEDOUT' || code === 'ESOCKETTIMEDOUT' || code === 'ECONNRESET') return true
  const msg = String(e?.message ?? '')
  if (/timeout|timed out|ETIMEDOUT|ESOCKETTIMEDOUT/i.test(msg)) return true
  if (/\b504\b/.test(msg)) return true
  return false
}

async function aiStreamTextWithIdleTimeout(params: {
  model: any
  system?: string
  messages: CoreMessage[]
  temperature?: number
  topP?: number
  providerOptions?: any
  idleTimeoutMs: number
  parentSignal?: AbortSignal
}): Promise<{ text: string }> {
  const { model, system, messages, temperature, topP, providerOptions, idleTimeoutMs, parentSignal } = params
  const ac = new AbortController()
  let timer: any
  let accumulatedText = ''
  let streamStarted = false

  const onParentAbort = () => { try { ac.abort() } catch { } }
  if (parentSignal) {
    if (parentSignal.aborted) throw new Error('Aborted by parent signal')
    parentSignal.addEventListener('abort', onParentAbort, { once: true })
  }

  const resetWatchdog = () => {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => { try { ac.abort() } catch { } }, idleTimeoutMs)
  }

  try {
    resetWatchdog()
    const callSignal = mergeAbortSignals([ac.signal, parentSignal])
    const result = streamText({
      model, system, messages, temperature, topP, providerOptions,
      abortSignal: callSignal, maxRetries: 0
    })
    streamStarted = true
    for await (const part of (result as any).fullStream as AsyncIterable<any>) {
      if (part?.type === 'text-delta') {
        resetWatchdog()
        accumulatedText += String(part.textDelta ?? '')
      }
    }
    clearTimeout(timer)
    return { text: accumulatedText }
  }
  catch (e: any) {
    clearTimeout(timer)
    if (parentSignal?.aborted) throw e
    if (ac.signal.aborted) {
      const kind = streamStarted ? 'stream_idle_timeout' : 'connection_timeout'
      throw new PlannerTimeoutError(idleTimeoutMs, { kind, original: e })
    }
    if (isLikelySdkOrNetworkTimeout(e))
      throw new PlannerTimeoutError(idleTimeoutMs, { kind: 'network_error', original: e })
    throw e
  }
  finally {
    clearTimeout(timer)
    parentSignal?.removeEventListener?.('abort', onParentAbort as any)
  }
}

async function aiGenerateJsonWithTimeoutRetry(params: {
  model: any; system: string; user: string; timeoutsMs?: number[]; parentSignal?: AbortSignal
  providerOptions?: any
  validator?: (text: string) => void | Promise<void>
  onTimeoutRetry?: (info: { attempt: number; timeoutMs: number; nextAttempt: number; nextTimeoutMs: number; reason: string; error: any }) => void
}): Promise<string> {
  const { model, system, user, parentSignal, onTimeoutRetry, validator, providerOptions } = params
  const timeoutsMsRaw = (params.timeoutsMs?.length ? params.timeoutsMs : [20_000, 30_000, 40_000]).map(n => Number(n)).filter(n => Number.isFinite(n) && n > 0)
  const timeoutsMs = timeoutsMsRaw.length ? timeoutsMsRaw : [20_000]
  const totalAttempts = timeoutsMs.length
  let lastErr: any
  for (let attempt = 1; attempt <= totalAttempts; attempt++) {
    const idleTimeoutMs = timeoutsMs[attempt - 1]
    try {
      const { text } = await aiStreamTextWithIdleTimeout({
        model, system, messages: [{ role: 'user', content: user }],
        temperature: 0, providerOptions, idleTimeoutMs, parentSignal,
      })
      if (validator) await validator(text)
      return text
    }
    catch (e: any) {
      lastErr = e
      if (parentSignal?.aborted) throw e
      if (attempt >= totalAttempts) throw e
      const nextAttempt = attempt + 1
      const nextTimeoutMs = timeoutsMs[nextAttempt - 1]
      const reason = summarizePlannerTimeoutReason(e)
      onTimeoutRetry?.({ attempt, timeoutMs: idleTimeoutMs, nextAttempt, nextTimeoutMs, reason, error: e })
      await sleep(250 * Math.pow(2, attempt - 1))
    }
  }
  throw lastErr
}

function parseTimeoutScheduleToMs(input: any): number[] | null {
  if (!input) return null
  const s = String(input).trim()
  if (!s) return null
  const parts = s.split(/[,，]/).map(x => x.trim()).filter(Boolean)
  const nums = parts.map(p => Number(p)).filter(n => Number.isFinite(n) && n > 0)
  if (!nums.length) return null
  const allLooksLikeSeconds = nums.every(n => n <= 300)
  const out = allLooksLikeSeconds ? nums.map(n => Math.round(n * 1000)) : nums.map(n => Math.round(n))
  return out.filter(n => n > 0)
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
    else if (Array.isArray(msg.text)) content = msg.text.map((p: any) => p?.type === 'text' ? p.text : '[File]').join('')
    else content = '[Complex Content]'
    messages.push(`${role}: ${content}`)
    currentId = msg.parentMessageId
  }
  return messages.reverse().join('\n')
}

function formatSearchRoundsForPlanner(rounds: SearchRound[]): string {
  if (!rounds.length) return '（无）'
  return rounds.map((r, idx) => {
    const items = (r.items || []).map((it, i) =>
      `- [${idx + 1}.${i + 1}] ${String(it.title || '').trim()}\n  ${String(it.url || '').trim()}\n  内容: ${String(it.content || '').replace(/\s+/g, ' ').trim()}`
    ).join('\n\n')
    const note = r.note ? `\n（注：${r.note}）` : ''
    return `### 第${idx + 1}轮 query="${r.query}"\n${items || '（无结果）'}${note}`
  }).join('\n\n')
}

async function planNextSearchAction(params: {
  plannerModel: any; userQuestion: string; rounds: SearchRound[]; fullContext: string
  priorContextSummary: string | null; date: string; abortSignal?: AbortSignal
  timeoutScheduleMs?: number[]
  onTimeoutRetry?: (info: { attempt: number; timeoutMs: number; nextAttempt: number; nextTimeoutMs: number; reason: string; error: any }) => void
}): Promise<SearchPlan> {
  const { plannerModel, userQuestion, rounds, fullContext, priorContextSummary, date, abortSignal, timeoutScheduleMs, onTimeoutRetry } = params
  const isFirstRound = !priorContextSummary
  const plannerSystem = [
    '你是"联网搜索规划器 & 结果筛选器"。你的任务是协助回答用户问题。', '',
    `当前时间：${date}`, '',
    '任务：', '1. 首先判断是否需要联网搜索。**这是最重要的决策。**',
    '2. 评估"已进行的搜索与结果"，选出对回答用户问题有价值的条目ID (格式如 "1.1", "2.3")。', '',
    '【决策逻辑】', '- **必须搜索**：问题涉及近期新闻、特定事实查证（天气、股价等）、非常识性具体数据，且上下文中没有。',
    '- **禁止搜索 (action="stop")**：', '  - 纯闲聊（如"你好"、"你是谁"）。', '  - 常识性问题。', '  - 逻辑推理、数学计算、代码编写、翻译、润色任务。',
    '  - **上下文已包含答案**。', '',
    '如果决定搜索：action="search"，并给出 query。',
    '如果无需搜索或信息已足：action="stop"。', '',
    isFirstRound
      ? '3. 请务必阅读完整的对话上下文，并生成一个精炼的"context_summary"（100-200字）。'
      : '3. 参考提供的 "context_summary" 来理解用户意图。', '',
    '输出严格 JSON：', '{', '  "action": "search" | "stop",', '  "query": string,', '  "reason": string,',
    '  "selected_ids": string[],', isFirstRound ? '  "context_summary": string' : null,
    '}'.replace(/, null/g, ''),
    '请只返回纯 JSON 字符串，不要包含 Markdown 格式。'
  ].filter(Boolean).join('\n')

  const contextBlock = isFirstRound
    ? `【完整对话上下文】\n${fullContext || '（无）'}`
    : `【历史上下文总结】\n${priorContextSummary}`

  const plannerUser = ['【用户问题】', userQuestion, '', contextBlock, '', '【已进行的搜索与结果】',
    formatSearchRoundsForPlanner(rounds), '', '现在请决定：selected_ids 是哪些？是否需要继续搜索？' + (isFirstRound ? ' 记得生成 context_summary。' : '')
  ].join('\n')

  const schedule = (timeoutScheduleMs && timeoutScheduleMs.length ? timeoutScheduleMs : [20_000, 30_000, 40_000])
  const jsonText = await aiGenerateJsonWithTimeoutRetry({
    model: plannerModel, system: plannerSystem, user: plannerUser, timeoutsMs: schedule, parentSignal: abortSignal,
    validator: async (text) => {
      const parsed = safeParseJsonFromText(text)
      if (!parsed) throw new Error(`JSON parsing failed`)
      if (!parsed.action) throw new Error('Invalid JSON: missing "action" field')
    },
    onTimeoutRetry,
  })
  return safeParseJsonFromText(jsonText) as SearchPlan
}

async function runIterativeWebSearch(params: {
  plannerModel: any; userQuestion: string; maxRounds: number; maxResults: number
  abortSignal?: AbortSignal; provider?: 'searxng' | 'tavily'; searxngApiUrl?: string; tavilyApiKey?: string
  onProgress?: (status: string) => void; fullContext: string; date: string; plannerTimeoutScheduleMs?: number[]
}): Promise<{ rounds: SearchRound[]; selectedIds: Set<string>; planFailed: boolean }> {
  const { plannerModel, userQuestion, maxRounds, maxResults, abortSignal, provider, searxngApiUrl, tavilyApiKey, onProgress, fullContext, date, plannerTimeoutScheduleMs } = params
  const rounds: SearchRound[] = []
  const usedQueries = new Set<string>()
  const selectedIds = new Set<string>()
  let currentContextSummary: string | null = null
  const schedule = (plannerTimeoutScheduleMs && plannerTimeoutScheduleMs.length ? plannerTimeoutScheduleMs : [20_000, 30_000, 40_000])
  let planFailed = false

  for (let i = 0; i < maxRounds; i++) {
    if (i === 0) onProgress?.('⏳ 正在分析用户意图，规划搜索策略...')
    let plan: SearchPlan | null = null
    let lastPlanError: any = null
    try {
      plan = await planNextSearchAction({
        plannerModel, userQuestion, rounds, abortSignal, fullContext, priorContextSummary: currentContextSummary, date,
        timeoutScheduleMs: schedule,
        onTimeoutRetry: ({ attempt, timeoutMs, nextAttempt, nextTimeoutMs, reason }) => {
          onProgress?.(`⚠️ 规划器第 ${attempt} 次请求失败 >${Math.round(timeoutMs / 1000)}s，正在重试第 ${nextAttempt} 次...\n   原因：${reason}`)
        },
      })
    }
    catch (e) { lastPlanError = e; if (API_DEBUG) debugLog('[SearchPlanner] failed:', (e as any)?.message ?? e) }

    if (!plan) {
      planFailed = true
      const reason = lastPlanError ? summarizeAnyError(lastPlanError instanceof PlannerTimeoutError ? lastPlanError.original ?? lastPlanError : lastPlanError) : ''
      onProgress?.(`❌ 规划服务暂时不可用。${reason ? `\n   原因：${reason}` : ''}`)
      break
    }
    if (plan.context_summary && typeof plan.context_summary === 'string') currentContextSummary = plan.context_summary
    if (Array.isArray(plan.selected_ids)) plan.selected_ids.forEach(id => selectedIds.add(String(id).trim()))
    const reasonText = plan.reason ? `(理由: ${plan.reason})` : ''
    if (plan.action !== 'search') {
      if (i === 0) onProgress?.(`🛑 模型判断无需搜索 ${reasonText}`)
      else onProgress?.(`✅ 信息收集完毕 ${reasonText}`)
      break
    }
    const q = String(plan.query || '').trim()
    if (!q) break
    if (usedQueries.has(q)) { onProgress?.(`⚠️ 关键词「${q}」已使用过，跳过重复搜索...`); break }
    usedQueries.add(q)
    onProgress?.(`🔍 正在搜索：「${q}」\n   🧠 ${reasonText}`)
    try {
      const r = await webSearch(q, { maxResults, signal: abortSignal, provider, searxngApiUrl, tavilyApiKey })
      const items = (r.results || []).slice(0, maxResults).map(it => ({
        title: String(it.title || ''), url: String(it.url || ''), content: removeImagesFromText(String(it.content || '')),
      }))
      rounds.push({ query: q, items })
      onProgress?.(`📄 搜索成功，获取到 ${items.length} 个页面，正在判断是否需要进一步搜索...`)
    }
    catch (e: any) {
      const errMsg = e?.message ?? String(e)
      console.error(`[WebSearch][Round ${i + 1}] Search failed for query "${q}":`, errMsg)
      rounds.push({ query: q, items: [], note: errMsg })
      onProgress?.(`❌ 搜索请求失败：${errMsg}`)
      break
    }
  }
  return { rounds, selectedIds, planFailed }
}

function formatAggregatedSearchForAnswer(rounds: SearchRound[]): string {
  if (!rounds.length) return ''
  let n = 0
  const lines: string[] = []
  const refLines: string[] = []
  lines.push('【联网搜索结果（已筛选）】')
  rounds.forEach((r) => {
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
  lines.push('', '【参考来源列表】', ...refLines, '', '【回答要求】',
    '- 基于以上来源回答用户问题。', '- 引用格式必须为 markdown 链接：[编号](url)。',
    '- 在回答末尾列出所有引用过的参考来源。', '- 若来源不足以支持结论，请明确说明。')
  return lines.join('\n')
}

function appendTextToMessageContent(content: MessageContent, appendix: string): MessageContent {
  if (!appendix?.trim()) return content
  if (typeof content === 'string') return `${content}\n\n${appendix}`
  if (Array.isArray(content)) {
    const idx = (content as any[]).findIndex(p => p?.type === 'text' && typeof p?.text === 'string')
    if (idx >= 0) {
      const arr = [...(content as any[])]
      arr[idx] = { ...(arr[idx] || {}), text: `${arr[idx].text}\n\n${appendix}` }
      return arr as any
    }
    return [{ type: 'text', text: appendix }, ...(content as any[])] as any
  }
  return content
}

const ErrorCodeMessage: Record<string, string> = {
  401: '提供错误的API密钥', 403: '服务器拒绝访问', 502: '错误的网关',
  503: '服务器繁忙，请稍后再试', 504: '网关超时', 500: '服务器繁忙，请稍后再试',
}

let auditService: TextAuditService
const _lockedKeys: { key: string; lockedTime: number }[] = []

const DATA_URL_IMAGE_CAPTURE_RE = /data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)/g
const MARKDOWN_IMAGE_RE = /!$$[^$$]*\]\s*$[^)]+?$\s*$/g
const UPLOADS_URL_RE = /(\/uploads\/[^)\s>"']+\.(?:png|jpe?g|webp|gif|bmp|heic))/gi
const HTML_IMAGE_RE = /<img[^>]*\ssrc=["']([^"']+)["'][^>]*>/gi

async function replaceDataUrlImagesWithUploads(text: string): Promise<{ text: string; saved: Array<{ mime: string; filename: string; bytes: number }> }> {
  if (!text) return { text: '', saved: [] }
  const saved: Array<{ mime: string; filename: string; bytes: number }> = []
  let out = ''
  let lastIndex = 0
  const matches = text.matchAll(DATA_URL_IMAGE_CAPTURE_RE)
  for (const m of matches) {
    const full = m[0]; const mime = m[1]; const base64 = m[2]; const idx = m.index ?? -1
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

type SavedUpload = { mime: string; filename: string; bytes: number }
type DataUrlCache = Map<string, string>

function parseDataUrlImage(dataUrl: string): { mime: string; base64: string } | null {
  const m = /^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$/.exec(dataUrl)
  if (!m) return null
  return { mime: m[1] || 'image/png', base64: (m[2] || '').replace(/\s+/g, '') || null } as any
}
function mimeToExt(mime: string): string {
  const t = (mime || '').toLowerCase()
  if (t === 'image/jpeg' || t === 'image/jpg') return 'jpg'
  if (t === 'image/png') return 'png'
  if (t === 'image/webp') return 'webp'
  if (t === 'image/gif') return 'gif'
  if (t === 'image/bmp') return 'bmp'
  if (t === 'image/heic') return 'heic'
  return t.split('/')[1] || 'png'
}

async function saveDataUrlImageToUploads(dataUrl: string, cache?: DataUrlCache): Promise<{ url: string; saved: SavedUpload } | null> {
  try {
    if (!dataUrl?.startsWith('data:image/')) return null
    if (cache?.has(dataUrl)) { const url = cache.get(dataUrl)!; return { url, saved: { mime: 'image/*', filename: path.basename(url), bytes: 0 } } }
    const parsed = parseDataUrlImage(dataUrl)
    if (!parsed) return null
    await ensureUploadDir()
    const buffer = Buffer.from(parsed.base64, 'base64')
    const ext = mimeToExt(parsed.mime)
    const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
    const filePath = path.join(UPLOAD_DIR, filename)
    await fs.writeFile(filePath, buffer)
    const url = `/uploads/${filename}`
    cache?.set(dataUrl, url)
    return { url, saved: { mime: parsed.mime, filename, bytes: buffer.length } }
  }
  catch (e) { globalThis.console.error('[saveDataUrlImageToUploads] failed:', e); return null }
}

async function normalizeMessageContentDataUrlsToUploads(content: MessageContent, cache?: DataUrlCache): Promise<{ content: MessageContent; saved: SavedUpload[] }> {
  const savedAll: SavedUpload[] = []
  if (typeof content === 'string') {
    const replaced = await replaceDataUrlImagesWithUploads(content)
    savedAll.push(...replaced.saved)
    return { content: replaced.text, saved: savedAll }
  }
  if (Array.isArray(content)) {
    const newParts: any[] = []
    for (const p of content as any[]) {
      if (p?.type === 'text' && typeof p.text === 'string') {
        const replaced = await replaceDataUrlImagesWithUploads(p.text)
        savedAll.push(...replaced.saved)
        newParts.push({ ...p, text: replaced.text })
        continue
      }
      if (p?.type === 'image_url' && typeof p.image_url?.url === 'string') {
        const u = p.image_url.url as string
        if (u.startsWith('data:image/')) {
          const r = await saveDataUrlImageToUploads(u, cache)
          if (r?.url) { savedAll.push(r.saved); newParts.push({ ...p, image_url: { ...p.image_url, url: r.url } }); continue }
        }
        newParts.push(p); continue
      }
      newParts.push(p)
    }
    return { content: newParts as any, saved: savedAll }
  }
  return { content, saved: savedAll }
}

async function normalizeUrlsDataUrlsToUploads(urls: string[], cache?: DataUrlCache): Promise<{ urls: string[]; saved: SavedUpload[] }> {
  const out: string[] = []; const savedAll: SavedUpload[] = []
  for (const u of urls || []) {
    if (typeof u === 'string' && u.startsWith('data:image/')) {
      const r = await saveDataUrlImageToUploads(u, cache)
      if (r?.url) { out.push(r.url); savedAll.push(r.saved) } else out.push(u)
    } else out.push(u)
  }
  return { urls: out, saved: savedAll }
}

function shortUrlForLog(u: string): string {
  if (!u) return ''
  if (u.startsWith('data:image/')) return `dataUrl(${u.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,/)?.[1] ?? 'image/*'},len=${u.length})`
  return u.length > 200 ? `${u.slice(0, 200)}...(len=${u.length})` : u
}

function extractImageUrlsFromText(text: string): { cleanedText: string; urls: string[] } {
  if (!text) return { cleanedText: '', urls: [] }
  const urls: string[] = []
  let cleaned = text
  cleaned = cleaned.replace(MARKDOWN_IMAGE_RE, (_m, inside) => { const raw = String(inside ?? '').trim(); if (raw) { const url = raw.split(/\s+/)[0]?.replace(/^<|>$/g, '').trim(); if (url) urls.push(url) }; return '[Image]' })
  cleaned = cleaned.replace(HTML_IMAGE_RE, (_m, url) => { if (url) urls.push(url); return '[Image]' })
  const rawDataUrls = cleaned.match(DATA_URL_IMAGE_RE)
  if (rawDataUrls?.length) urls.push(...rawDataUrls)
  cleaned = cleaned.replace(DATA_URL_IMAGE_RE, '[Image]')
  const uploadUrls = cleaned.match(UPLOADS_URL_RE)
  if (uploadUrls?.length) urls.push(...uploadUrls)
  cleaned = cleaned.replace(UPLOADS_URL_RE, '[Image]')
  return { cleanedText: cleaned, urls }
}

async function extractImageUrlsFromMessageContent(content: MessageContent): Promise<{ cleanedText: string; urls: string[] }> {
  const urls: string[] = []; let cleanedText = ''
  if (typeof content === 'string') { const r = extractImageUrlsFromText(content); cleanedText = r.cleanedText; urls.push(...r.urls); if (!cleanedText?.trim() && urls.length) cleanedText = '[Image]'; return { cleanedText, urls } }
  if (Array.isArray(content)) {
    for (const p of content as any[]) {
      if (p?.type === 'text' && p.text) { const r = extractImageUrlsFromText(p.text); if (r.cleanedText) cleanedText += (cleanedText ? '\n' : '') + r.cleanedText; if (r.urls.length) urls.push(...r.urls) }
      else if (p?.type === 'image_url' && p.image_url?.url) { urls.push(p.image_url.url); cleanedText += (cleanedText ? '\n' : '') + '[Image]' }
    }
    if (!cleanedText?.trim() && urls.length) cleanedText = '[Image]'
    return { cleanedText, urls }
  }
  return { cleanedText: '', urls: [] }
}

function dedupeUrlsPreserveOrder(urls: string[]): string[] {
  const seen = new Set<string>(); const out: string[] = []
  for (const u of urls) { if (!u || seen.has(u)) continue; seen.add(u); out.push(u) }
  return out
}

type ImageBinding = { tag: string; source: 'current_user' | 'last_ai' | 'prev_user' | 'prev_ai'; url: string }
type HistoryMessageMeta = { role: 'user' | 'model'; cleanedText: string; urls: string[]; messageId: string }

function applyImageBindingsToCleanedText(cleanedText: string, urls: string[], bindings: ImageBinding[]): string {
  const map = new Map<string, string>()
  for (const b of bindings) map.set(b.url, b.tag)
  let out = cleanedText || ''
  for (const u of urls) { const tag = map.get(u); out = out.replace('[Image]', tag ? `[${tag}]` : '[Image]') }
  if (!out.trim() && urls.length) out = urls.map(u => (map.get(u) ? `[${map.get(u)}]` : '[Image]')).join(' ')
  return out
}

async function buildGeminiHistoryFromLastMessageId(params: { lastMessageId?: string; maxContextCount: number; forGeminiImageModel: boolean }): Promise<{ metas: HistoryMessageMeta[]; allUrls: string[] }> {
  const { lastMessageId: startId, maxContextCount, forGeminiImageModel } = params
  const metasRev: HistoryMessageMeta[] = []; const urlsRev: string[] = []
  let lastMessageId = startId
  for (let i = 0; i < maxContextCount; i++) {
    if (!lastMessageId) break; const msg = await getMessageById(lastMessageId); if (!msg) break
    const role = (msg.role === 'assistant' ? 'model' : 'user') as 'user' | 'model'
    if (forGeminiImageModel) {
      const { cleanedText, urls } = await extractImageUrlsFromMessageContent(msg.text)
      if (urls.length) urlsRev.push(...urls)
      metasRev.push({ role, cleanedText: cleanedText?.trim() ? cleanedText : (urls.length ? '[Image]' : '[Empty]'), urls, messageId: msg.id })
    } else {
      metasRev.push({ role, cleanedText: typeof msg.text === 'string' ? msg.text : '[Complex Content]', urls: [], messageId: msg.id })
    }
    lastMessageId = msg.parentMessageId
  }
  return { metas: metasRev.reverse(), allUrls: dedupeUrlsPreserveOrder(urlsRev.reverse()) }
}

function getLastNByRole(metas: HistoryMessageMeta[], role: 'user' | 'model', n: number): HistoryMessageMeta[] {
  const out: HistoryMessageMeta[] = []
  for (let i = metas.length - 1; i >= 0; i--) { if (metas[i].role !== role) continue; out.push(metas[i]); if (out.length >= n) break }
  return out
}

function selectRecentTwoRoundsImages(metas: HistoryMessageMeta[], currentUrls: string[]): ImageBinding[] {
  const lastModels = getLastNByRole(metas, 'model', 2); const lastUsers = getLastNByRole(metas, 'user', 1)
  const groups = [
    { source: 'current_user' as const, tagPrefix: 'U0', urls: currentUrls },
    { source: 'last_ai' as const, tagPrefix: 'A0', urls: lastModels[0]?.urls ?? [] },
    { source: 'prev_user' as const, tagPrefix: 'U1', urls: lastUsers[0]?.urls ?? [] },
    { source: 'prev_ai' as const, tagPrefix: 'A1', urls: lastModels[1]?.urls ?? [] },
  ]
  const seen = new Set<string>(); const bindings: ImageBinding[] = []
  for (const g of groups) { let idx = 0; for (const u of g.urls) { if (!u || seen.has(u)) continue; seen.add(u); idx += 1; bindings.push({ tag: `${g.tagPrefix}_${idx}`, source: g.source, url: u }) } }
  return bindings
}

const IMAGE_SOURCE_LABEL: Record<ImageBinding['source'], string> = {
  current_user: '本次用户消息', last_ai: '最近一条AI回复', prev_user: '上一次用户消息', prev_ai: '倒数第二条AI回复',
}

type AiPart = { type: 'text'; text: string } | { type: 'image'; image: string } | { inlineData: { mimeType: string; data: string } }

async function imageUrlToAiSdkImagePart(urlStr: string): Promise<AiPart | null> {
  if (!urlStr) return null
  if (urlStr.startsWith('data:image/')) {
    const match = urlStr.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/)
    if (match) return { inlineData: { mimeType: match[1], data: match[2] } }
    return { type: 'image', image: urlStr }
  }
  if (urlStr.includes('/uploads/')) {
    const fileData = await getFileBase64(path.basename(urlStr))
    if (!fileData) return null
    return { inlineData: { mimeType: fileData.mime, data: fileData.data } }
  }
  return null
}

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

  const existingIdx = processThreads.findIndex(t => t.key === threadKey)
  if (existingIdx > -1) { processThreads[existingIdx].abort.abort(); processThreads.splice(existingIdx, 1) }
  processThreads.push({ key: threadKey, userId, abort, messageId, roomId })

  await ensureUploadDir()

  const model = options.room.chatModel
  const maxContextCount = options.user.advanced.maxContextCount ?? 20
  updateRoomChatModel(userId, options.room.roomId, model)

  const { message, uploadFileKeys, lastContext, process: processCb, systemMessage, temperature, top_p } = options

  processCb?.({ id: customMessageId, text: '', role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })

  let content: MessageContent = message
  let fileContext = ''

  const globalConfig = await getCacheConfig()
  const imageModelsStr = globalConfig.siteConfig.imageModels || ''
  const imageModelList = imageModelsStr.split(/[,，]/).map(s => s.trim()).filter(Boolean)
  const isImage = imageModelList.some(m => model.includes(m))

  if (uploadFileKeys && uploadFileKeys.length > 0) {
    const textFiles = uploadFileKeys.filter(k => isTextFile(k))
    const imageFiles = uploadFileKeys.filter(k => isImageFile(k))
    const multiModalFiles = uploadFileKeys.filter(k => isMultiModalFile(k))

    if (textFiles.length > 0) {
      for (const fileKey of textFiles) {
        try {
          const filePath = path.join(UPLOAD_DIR, stripTypePrefix(fileKey))
          await fs.access(filePath)
          const fileContent = await fs.readFile(filePath, 'utf-8')
          fileContext += `\n\n--- File Start: ${stripTypePrefix(fileKey)} ---\n${fileContent}\n--- File End ---\n`
        }
        catch (e) {
          globalThis.console.error(`Error reading text file ${fileKey}`, e)
          fileContext += `\n\n[System Error: File ${fileKey} not found or unreadable]\n`
        }
      }
    }

    const finalMessage = message + (fileContext ? `\n\nAttached Files content:\n${fileContext}` : '')

    if (imageFiles.length > 0 || multiModalFiles.length > 0) {
      content = [{ type: 'text', text: finalMessage }] as any
      for (const uploadFileKey of imageFiles) {
        ;(content as any[]).push({ type: 'image_url', image_url: { url: await convertImageUrl(stripTypePrefix(uploadFileKey)) } })
      }
      for (const uploadFileKey of multiModalFiles) {
        const fileBase64 = await getFileBase64(uploadFileKey)
        if (fileBase64) {
          ;(content as any[]).push({ type: 'file', mimeType: fileBase64.mime, data: fileBase64.data })
        }
      }
    }
    else {
      content = finalMessage
    }
  }

  let searchProcessLog = ''
  const allowSearch = globalConfig.siteConfig?.webSearchEnabled === true
  const finalSearchMode = allowSearch && options.searchMode === true && !isImage

  if (finalSearchMode) {
    try {
      const preContext = await loadContextMessages({ maxContextCount, lastMessageId: lastContext?.parentMessageId, systemMessage, content })
      const preTokens = estimateTokenCount(preContext)
      const plannerModelCfg = String(globalConfig.siteConfig?.webSearchPlannerModel ?? '').trim()
      const plannerModelEnv = String(process.env.WEB_SEARCH_PLANNER_MODEL ?? '').trim()
      const plannerModelName = plannerModelCfg || plannerModelEnv || model
      let plannerKey = await getRandomApiKey(options.user, model, undefined, preTokens)
      if (!plannerKey) throw new Error('没有对应的 apikeys 配置 (Planner)')
      let actualPlannerModel = plannerModelName
      if (plannerModelName !== model) {
        const candidateKey = await getRandomApiKey(options.user, plannerModelName, undefined, preTokens)
        if (candidateKey) plannerKey = candidateKey
        else actualPlannerModel = model
      }
      const plannerProvider = normalizeProvider((plannerKey as any).keyModel)
      const plannerBaseUrl = isNotEmptyString(plannerKey.baseUrl) ? plannerKey.baseUrl : (isNotEmptyString(globalConfig.apiBaseUrl) ? globalConfig.apiBaseUrl : undefined)
      const maxRounds = Math.max(1, Math.min(6, Number(globalConfig.siteConfig?.webSearchMaxRounds ?? process.env.WEB_SEARCH_MAX_ROUNDS ?? 3)))
      const maxResults = Math.max(1, Math.min(10, Number(globalConfig.siteConfig?.webSearchMaxResults ?? process.env.WEB_SEARCH_MAX_RESULTS ?? 5)))
      const searchProvider = globalConfig.siteConfig?.webSearchProvider as any
      const searchSearxngUrl = String(globalConfig.siteConfig?.searxngApiUrl ?? '').trim() || undefined
      const searchTavilyKey = String(globalConfig.siteConfig?.tavilyApiKey ?? '').trim() || undefined
      const progressMessages: string[] = []
      const onProgressLocal = (status: string) => {
        progressMessages.push(status); searchProcessLog = progressMessages.join('\n')
        const isFinishing = status.includes('完毕') || status.includes('无需')
        const suffix = isFinishing ? '\n\n⚡️ 准备生成...' : '\n\n⏳ 正在执行...'
        processCb?.({ id: customMessageId, text: searchProcessLog + suffix, role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })
      }
      const historyContextStr = await buildConversationContext(lastContext?.parentMessageId, maxContextCount)
      const currentDate = new Date().toLocaleString()
      const plannerTimeoutScheduleMs = parseTimeoutScheduleToMs((globalConfig.siteConfig as any)?.webSearchPlannerTimeoutScheduleMs ?? process.env.WEB_SEARCH_PLANNER_TIMEOUT_SCHEDULE_MS) ?? [20_000, 30_000, 40_000]
      const plannerModel = createLanguageModel({ provider: plannerProvider, apiKey: plannerKey.key, baseUrl: plannerBaseUrl, model: actualPlannerModel })
      const { rounds, selectedIds, planFailed } = await runIterativeWebSearch({
        plannerModel, userQuestion: message, maxRounds, maxResults, abortSignal: abort.signal,
        provider: searchProvider, searxngApiUrl: searchSearxngUrl, tavilyApiKey: searchTavilyKey,
        onProgress: onProgressLocal, fullContext: historyContextStr, date: currentDate, plannerTimeoutScheduleMs,
      })
      if (progressMessages.length > 0) searchProcessLog = progressMessages.join('\n')
      const filteredRounds = rounds.map((r, rIdx) => ({
        query: r.query, note: r.note,
        items: planFailed ? r.items : r.items.filter((_, iIdx) => selectedIds.has(`${rIdx + 1}.${iIdx + 1}`)),
      })).filter(r => r.items.length > 0 || r.note)
      const ctx = formatAggregatedSearchForAnswer(filteredRounds)
      let finalStatusMessage = '✅ 资料整理完毕，正在生成回答...'
      if (planFailed && rounds.length > 0) finalStatusMessage = '⚠️ 资料整理发生异常，保留现存搜索资料进行回答...'
      else if (!ctx && rounds.length > 0) finalStatusMessage = '⚠️ 未能筛选出有效引用，尝试直接回答...'
      processCb?.({ id: customMessageId, text: (searchProcessLog ? searchProcessLog + '\n\n---\n\n' : '') + `⚡️ ${finalStatusMessage}`, role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })
      if (ctx) content = appendTextToMessageContent(content, ctx)
    }
    catch (e: any) {
      if (isAbortError(e, abort.signal)) throw e
      globalThis.console.error('[WebSearch] failed:', e?.message ?? e)
      searchProcessLog += `\n❌ 联网搜索模块遇到问题：${e?.message ?? '未知错误'}`
      processCb?.({ id: customMessageId, text: searchProcessLog + '\n\n尝试使用已有知识回答...', role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })
    }
  }

  const finalMessages = await loadContextMessages({ maxContextCount, lastMessageId: lastContext?.parentMessageId, systemMessage, content })
  const finalTokenCount = estimateTokenCount(finalMessages)
  const key = await getRandomApiKey(options.user, model, undefined, finalTokenCount)

  try {
    if (!key) throw new Error(`没有对应的 apikeys 配置 (Token: ${finalTokenCount} / Model: ${model})`)
    const resolvedBaseUrl = isNotEmptyString(key.baseUrl) ? key.baseUrl : (isNotEmptyString(globalConfig.apiBaseUrl) ? globalConfig.apiBaseUrl : undefined)
    const provider = normalizeProvider((key as any).keyModel)
    const lm = createLanguageModel({ provider, apiKey: key.key, baseUrl: resolvedBaseUrl, model })
    const modelConfig = MODEL_CONFIGS[model] || { supportTopP: true }
    const finalTemperature = modelConfig.defaultTemperature ?? temperature
    const shouldUseTopP = modelConfig.supportTopP

    const getCombineStreamText = (currentAnswerText: string) => {
      if (!searchProcessLog) return currentAnswerText
      return `${searchProcessLog}\n\n---\n\n${currentAnswerText}`
    }

    const isGeminiImageModel = isImage && provider === 'google' && model.includes('gemini')

    if (isGeminiImageModel) {
      const dataUrlCache: DataUrlCache = new Map()
      const normalizedCurrent = await normalizeMessageContentDataUrlsToUploads(content, dataUrlCache)
      content = normalizedCurrent.content
      const { metas } = await buildGeminiHistoryFromLastMessageId({ lastMessageId: lastContext?.parentMessageId, maxContextCount, forGeminiImageModel: true })
      const metasNormalized: HistoryMessageMeta[] = []
      for (const m of metas) { const nu = await normalizeUrlsDataUrlsToUploads(m.urls || [], dataUrlCache); metasNormalized.push({ ...m, urls: nu.urls }) }
      const { cleanedText: currentCleanedText, urls: currentUrlsRaw } = await extractImageUrlsFromMessageContent(content)
      const currentUrlsNorm = await normalizeUrlsDataUrlsToUploads(currentUrlsRaw || [], dataUrlCache)
      const currentUrls = currentUrlsNorm.urls
      const bindings = selectRecentTwoRoundsImages(metasNormalized, currentUrls)
      const historyNative = metasNormalized.map(m => ({ role: m.role === 'model' ? 'model' : 'user', parts: [{ text: applyImageBindingsToCleanedText(m.cleanedText, m.urls, bindings) || '[Empty]' }] }))
      const userInstructionBase = (currentCleanedText && currentCleanedText.trim()) ? currentCleanedText.trim() : '请继续生成/修改图片。'
      const labeledUserInstruction = applyImageBindingsToCleanedText(userInstructionBase, currentUrls, bindings)
      const nativeUserParts: any[] = []
      if (bindings.length) nativeUserParts.push({ text: `【最近两轮图片映射表】\n${bindings.map(b => `${b.tag}=${IMAGE_SOURCE_LABEL[b.source]}(${shortUrlForLog(b.url)})`).join('\n')}\n` })
      for (const b of bindings) {
        nativeUserParts.push({ text: `【${b.tag}｜${IMAGE_SOURCE_LABEL[b.source]}】` })
        const img = await imageUrlToAiSdkImagePart(b.url)
        if (img) { if ('inlineData' in img) nativeUserParts.push({ inlineData: img.inlineData }); else if (img.type === 'text') nativeUserParts.push({ text: img.text }) }
        else nativeUserParts.push({ text: `（该图片无法上传：${shortUrlForLog(b.url)}）` })
      }
      nativeUserParts.push({ text: `【编辑指令】${labeledUserInstruction}` })
      const callSignal = mergeAbortSignals([abort.signal, globalConfig.timeoutMs ? AbortSignal.timeout(globalConfig.timeoutMs) : undefined])
      let fetchUrl = `${resolvedBaseUrl ? resolvedBaseUrl.replace(/\/+$/, '') : 'https://generativelanguage.googleapis.com/v1beta'}`
      if (!fetchUrl.includes('/models/')) fetchUrl += `/models/${model}:generateContent`
      const isGemini3ProImageModel = model.includes('gemini-3-pro-image')
      const requestBody = {
        contents: [...historyNative, { role: 'user', parts: nativeUserParts }],
        generationConfig: { temperature: finalTemperature, topP: shouldUseTopP ? top_p : undefined,
          ...(isGemini3ProImageModel ? { responseModalities: ['TEXT', 'IMAGE'], imageConfig: { imageSize: '4K' } } : {}) },
      }
      const res = await fetch(fetchUrl, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key.key, 'Authorization': `Bearer ${key.key}` },
        body: JSON.stringify(requestBody), signal: callSignal,
      })
      if (!res.ok) { const errText = await res.text().catch(() => 'No Content'); throw new Error(`API Error (${res.status}): ${errText}`) }
      const jsonRes = await res.json()
      let answerText = ''; const outputParts = jsonRes.candidates?.[0]?.content?.parts || []
      for (const part of outputParts) {
        if (part.text) answerText += part.text
        if (part.inlineData && part.inlineData.data) {
          const mime = part.inlineData.mimeType || 'image/png'; const ext = mime.split('/')[1] || 'png'
          const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
          await fs.writeFile(path.join(UPLOAD_DIR, filename), Buffer.from(part.inlineData.data, 'base64'))
          answerText += `${answerText ? '\n\n' : ''}![Generated Image](/uploads/${filename})`
        }
      }
      if (!answerText) {
        if (jsonRes.promptFeedback?.blockReason) answerText = `[Gemini] 系统拦截了此请求，原因: ${jsonRes.promptFeedback.blockReason}`
        else { answerText = '[Gemini] No content returned.'; globalThis.console.error('Empty return:', JSON.stringify(jsonRes)) }
      }
      let usageRes: any = undefined
      if (jsonRes.usageMetadata) usageRes = { prompt_tokens: jsonRes.usageMetadata.promptTokenCount || 0, completion_tokens: jsonRes.usageMetadata.candidatesTokenCount || 0, total_tokens: jsonRes.usageMetadata.totalTokenCount || 0, estimated: false }
      processCb?.({ id: customMessageId, text: getCombineStreamText(answerText), role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: { usage: usageRes } })
      return sendResponse({ type: 'Success', data: { object: 'chat.completion', choices: [{ message: { role: 'assistant', content: answerText }, finish_reason: 'stop', index: 0, logprobs: null }], created: Date.now(), conversationId: lastContext?.conversationId, model, text: answerText, id: customMessageId, detail: { usage: usageRes } } })
    }

    const coreMessages = toCoreMessages(finalMessages)
    const callSignal = mergeAbortSignals([abort.signal, globalConfig.timeoutMs ? AbortSignal.timeout(globalConfig.timeoutMs) : undefined])

    if (isImage) {
      const result: any = await generateText({ model: lm, messages: coreMessages, temperature: finalTemperature, topP: shouldUseTopP ? top_p : undefined, abortSignal: callSignal, maxRetries: 0 })
      let answerText = result.text || ''
      if (result.files?.length) {
        for (const f of result.files) {
          if (!f?.mediaType?.startsWith('image/')) continue
          const ext = f.mediaType.split('/')[1] || 'png'
          const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}.${ext}`
          await fs.writeFile(path.join(UPLOAD_DIR, filename), Buffer.from(f.uint8Array))
          answerText += `${answerText ? '\n\n' : ''}![Generated Image](/uploads/${filename})`
        }
      }
      if (answerText && !answerText.startsWith('![') && (answerText.startsWith('http') || answerText.startsWith('data:image'))) answerText = `![Generated Image](${answerText})`
      const usageRes = toUsageResponse(result.usage)
      processCb?.({ id: customMessageId, text: getCombineStreamText(answerText), role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: { usage: usageRes } })
      return sendResponse({ type: 'Success', data: { object: 'chat.completion', choices: [{ message: { role: 'assistant', content: answerText }, finish_reason: 'stop', index: 0, logprobs: null }], created: Date.now(), conversationId: lastContext?.conversationId, model, text: answerText, id: customMessageId, detail: { usage: usageRes } } })
    }

    // ✅ providerOptions：对所有 OpenAI 类型（completions + responses）都传 reasoning_effort
    const result: any = streamText({
      model: lm, messages: coreMessages, temperature: finalTemperature, topP: shouldUseTopP ? top_p : undefined, abortSignal: callSignal, maxRetries: 0,
      providerOptions: isOpenAILike(provider)
        ? { openai: (() => { const siteCfg = globalConfig.siteConfig; const reasoningModelsStr = siteCfg?.reasoningModels || ''; const reasoningEffort = siteCfg?.reasoningEffort || 'medium'; const reasoningModelList = reasoningModelsStr.split(/[,，]/).map(s => s.trim()).filter(Boolean); if (reasoningModelList.includes(model) && reasoningEffort && reasoningEffort !== 'none') return { reasoning_effort: reasoningEffort }; return {} })() }
        : undefined,
    })

    let answerText = ''
    for await (const part of (result as any).fullStream as AsyncIterable<any>) {
      if (part?.type === 'text-delta') {
        const delta = String(part.textDelta ?? '')
        answerText += delta
        processCb?.({ id: customMessageId, text: getCombineStreamText(answerText), role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })
      }
      else if (part?.type === 'error') {
        globalThis.console.error('[Stream Error Part]', part.error)
        throw part.error
      }
    }

    let usageRes: any
    try { const usage = await (result.usage as Promise<any> | undefined); usageRes = toUsageResponse(usage) } catch { }

    return sendResponse({ type: 'Success', data: { object: 'chat.completion', choices: [{ message: { role: 'assistant', content: answerText }, finish_reason: 'stop', index: 0, logprobs: null }], created: Date.now(), conversationId: lastContext?.conversationId, model, text: answerText, id: customMessageId, detail: { usage: usageRes && { ...usageRes, estimated: false } } } })
  }
  catch (error: any) {
    if (isAbortError(error, abort.signal)) throw error

    const realError = error?.lastError ?? error?.cause ?? error
    const code = realError?.statusCode || realError?.status || realError?.response?.status || error?.statusCode

    let safeMessage = realError?.message ?? String(realError)
    if (realError?.responseBody) {
      try {
        const parsed = JSON.parse(realError.responseBody)
        if (parsed?.error?.message) safeMessage = parsed.error.message
      } catch {}
    }
    const errorMsgRaw = safeMessage

    let displayErrorMsg = errorMsgRaw
    if (code && Reflect.has(ErrorCodeMessage, code)) displayErrorMsg = `${ErrorCodeMessage[code]} (${errorMsgRaw})`
    else if (code) displayErrorMsg = `[Status ${code}] ${errorMsgRaw}`

    if ((code === 429 || code === 503) && (errorMsgRaw.includes('Too Many Requests') || errorMsgRaw.includes('Rate limit') || errorMsgRaw.includes('负载已饱和') || errorMsgRaw.includes('耗尽') || errorMsgRaw.includes('insufficient'))) {
      if (key && options.tryCount < 3) {
        options.tryCount++
        _lockedKeys.push({ key: key.key, lockedTime: Date.now() })
        await new Promise(resolve => setTimeout(resolve, 2000))
        const index = processThreads.findIndex(d => d.key === threadKey)
        if (index > -1) processThreads.splice(index, 1)
        
        try {
          return await chatReplyProcess(options)
        } catch (retryError: any) {
          if (String(retryError?.message).includes('没有对应的 apikeys 配置')) {
            processCb?.({ id: customMessageId, text: searchProcessLog ? `${searchProcessLog}\n\n---\n\n❌ 模型请求失败：${displayErrorMsg}` : `❌ 模型请求失败：${displayErrorMsg}`, role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })
            return Promise.reject({ message: displayErrorMsg, data: null, status: 'Fail' })
          }
          throw retryError
        }
      }
    }

    globalThis.console.error(`[ChatReply Process Error]:`, realError)

    const finalErrorText = searchProcessLog ? `${searchProcessLog}\n\n---\n\n❌ 模型请求失败：${displayErrorMsg}` : `❌ 模型请求失败：${displayErrorMsg}`
    processCb?.({ id: customMessageId, text: finalErrorText, role: 'assistant', conversationId: lastContext?.conversationId, parentMessageId: lastContext?.parentMessageId, detail: undefined })
    
    return Promise.reject({ message: displayErrorMsg, data: null, status: 'Fail' })
  }
  finally {
    const index = processThreads.findIndex(d => d.key === threadKey)
    if (index > -1) processThreads.splice(index, 1)
  }
}

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
    if (audit.sensitiveWords.split('\n').filter(d => textLower.includes(d.trim().toLowerCase())).length > 0) return true
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
  const parentMessageId = isPrompt ? chatInfo.options.parentMessageId : `prompt_${id}`
  if (chatInfo.status !== Status.Normal) return parentMessageId ? getMessageById(parentMessageId) : undefined

  if (isPrompt) {
    let promptText = chatInfo.prompt
    const allFileKeys = chatInfo.images || []
    const textFiles = allFileKeys.filter(k => isTextFile(k))
    const imageFiles = allFileKeys.filter(k => isImageFile(k))
    const multiModalFiles = allFileKeys.filter(k => isMultiModalFile(k))

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

    if (promptText && typeof promptText === 'string')
      promptText = promptText.replace(DATA_URL_IMAGE_RE, '[Image Data Removed]')

    let content: MessageContent = promptText
    if (imageFiles.length > 0 || multiModalFiles.length > 0) {
      content = [{ type: 'text', text: promptText }] as any
      for (const image of imageFiles) {
        ;(content as any[]).push({ type: 'image_url', image_url: { url: await convertImageUrl(stripTypePrefix(image)) } })
      }
      for (const fileKey of multiModalFiles) {
        const fileBase64 = await getFileBase64(fileKey)
        if (fileBase64) {
          ;(content as any[]).push({ type: 'file', mimeType: fileBase64.mime, data: fileBase64.data })
        }
      }
    }

    return { id, conversationId: chatInfo.options.conversationId, parentMessageId, role: 'user', text: content }
  }
  else {
    let responseText = chatInfo.response || ''
    if (responseText && typeof responseText === 'string')
      responseText = responseText.replace(DATA_URL_IMAGE_RE, '[Image History]')
    return { id, conversationId: chatInfo.options.conversationId, parentMessageId, role: 'assistant', text: responseText }
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

async function getRandomApiKey(user: UserInfo, chatModel: string, _accountId?: string, tokenCount?: number): Promise<KeyConfig | undefined> {
  let keys = (await getCacheApiKeys()).filter(d => hasAnyRole(d.userRoles, user.roles)).filter(d => d.chatModels.includes(chatModel))
  if (tokenCount !== undefined && keys.length > 0) {
    keys = keys.filter(k => {
      const remark = k.remark || ''
      const maxMatch = remark.match(/MAX_TOKENS?[:=]\s*(\d+)/i)
      const minMatch = remark.match(/MIN_TOKENS?[:=]\s*(\d+)/i)
      let valid = true
      if (maxMatch) { const max = parseInt(maxMatch[1], 10); if (tokenCount > max) valid = false }
      if (minMatch) { const min = parseInt(minMatch[1], 10); if (tokenCount <= min) valid = false }
      return valid
    })
  }
  const picked = await randomKeyConfig(keys)
  if (API_DEBUG) { debugLog('====== [Key Pick Debug] ======'); debugLog('[chatModel]', chatModel, '[tokens]', tokenCount); debugLog('[candidateKeyCount]', keys.length); debugLog('[picked]', picked ? { keyModel: (picked as any).keyModel, baseUrl: picked.baseUrl, remark: picked.remark } : '(null)'); debugLog('====== [Key Pick Debug End] ======') }
  return picked ?? undefined
}

export { chatReplyProcess, chatConfig, containsSensitiveWords }
