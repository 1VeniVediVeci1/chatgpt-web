// service/src/ai/provider.ts
import { createOpenAI } from '@ai-sdk/openai'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import nodeFetch from 'node-fetch'

/**
 * ✅ Debug 总开关：按你的要求“默认开启”，不再读取 .env
 * - API_DEBUG: 是否输出调试日志
 * - API_DEBUG_FETCH: 是否抓包输出上游 API 请求/响应
 */
const API_DEBUG = true
const API_DEBUG_FETCH = true

/**
 * 响应体预览最多读取多少字节
 * - 普通 JSON/文本：resp.clone().text() 后截断
 * - SSE(text/event-stream)：读取流的前 N bytes
 */
const API_DEBUG_FETCH_BODY_BYTES = 2048

export type KeyProvider = 'openai-compatible' | 'google'

export function normalizeProvider(keyModel: string | undefined): KeyProvider {
  const m = String(keyModel || '').trim()
  if (!m || m === 'ChatGPTAPI')
    return 'openai-compatible'
  if (m === 'ChatGPTUnofficialProxyAPI')
    return 'openai-compatible' // 兼容旧数据
  if (m === 'openai-compatible')
    return 'openai-compatible'
  if (m === 'google')
    return 'google'
  return 'openai-compatible'
}

function normalizeBaseUrl(baseUrl?: string) {
  const s = String(baseUrl || '').trim()
  if (!s) return undefined
  return s.replace(/\/+$/, '')
}

const baseFetch: any = (globalThis as any).fetch
  ? (globalThis as any).fetch.bind(globalThis)
  : (nodeFetch as any)

function trunc(s: any, n = 1200): string {
  const str = typeof s === 'string'
    ? s
    : (() => {
        try { return JSON.stringify(s) } catch { return String(s ?? '') }
      })()

  if (!str) return ''
  return str.length > n ? `${str.slice(0, n)}...(truncated,total=${str.length})` : str
}

function redactHeaders(headers: any) {
  const out: Record<string, any> = {}
  try {
    // Headers
    if (headers?.forEach) {
      headers.forEach((v: any, k: any) => { out[String(k)] = v })
    }
    // plain object
    else if (headers && typeof headers === 'object') {
      Object.assign(out, headers)
    }
  }
  catch {}

  for (const k of Object.keys(out)) {
    const lk = k.toLowerCase()
    if (
      lk === 'authorization'
      || lk.includes('api-key')
      || lk.includes('apikey')
      || lk.includes('x-api-key')
    ) {
      const v = String(out[k] ?? '')
      out[k] = v.length > 16 ? `${v.slice(0, 8)}***${v.slice(-4)}` : '***'
    }
  }
  return out
}

async function readStreamPreview(body: any, maxBytes: number) {
  try {
    if (!body?.getReader) return ''
    const reader = body.getReader()
    const chunks: Uint8Array[] = []
    let total = 0

    while (total < maxBytes) {
      const { value, done } = await reader.read()
      if (done) break
      if (value) {
        chunks.push(value)
        total += value.length
      }
    }

    // 只取消 clone 的分支，避免等待整个 SSE 完成
    try { await reader.cancel() } catch {}

    const buf = Buffer.concat(chunks.map(c => Buffer.from(c)))
    return buf.toString('utf8')
  }
  catch {
    return ''
  }
}

/**
 * ✅ 上游抓包：打印 Request/Response/Preview
 * - 不会泄露 Authorization（已脱敏）
 * - SSE 只取前 N bytes
 */
function wrapFetchWithDebug(fetchFn: any) {
  return async (input: any, init?: any) => {
    if (!API_DEBUG || !API_DEBUG_FETCH)
      return fetchFn(input, init)

    const url = typeof input === 'string'
      ? input
      : (input?.url ? String(input.url) : String(input))

    const method = String(init?.method ?? 'GET').toUpperCase()
    const headers = redactHeaders(init?.headers)

    const bodyPreview = (() => {
      const b = init?.body
      if (b == null) return ''
      if (typeof b === 'string') return trunc(b, 1200)
      return `[non-string body: ${Object.prototype.toString.call(b)}]`
    })()

    const start = Date.now()

    console.log('====== [Upstream Fetch Debug] ======')
    console.log('[Request]', { method, url, headers, bodyPreview })

    try {
      const resp = await fetchFn(input, init)
      const ms = Date.now() - start

      const ct = String(resp?.headers?.get?.('content-type') ?? '')
      const respHeaders = redactHeaders(resp?.headers)

      let preview = ''
      if (ct.includes('text/event-stream')) {
        // SSE：读取 clone 的前 N bytes
        preview = await readStreamPreview(resp.clone().body, API_DEBUG_FETCH_BODY_BYTES)
      }
      else {
        const txt = await resp.clone().text().catch(() => '')
        preview = txt ? trunc(txt, API_DEBUG_FETCH_BODY_BYTES) : ''
      }

      console.log('[Response]', {
        status: resp.status,
        statusText: resp.statusText,
        ms,
        contentType: ct,
        headers: respHeaders,
      })
      console.log('[ResponsePreview]', preview || '(empty)')
      console.log('====== [Upstream Fetch Debug End] ======')

      return resp
    }
    catch (e: any) {
      const ms = Date.now() - start
      console.log('[FetchError]', { ms, message: e?.message ?? String(e) })
      console.log('====== [Upstream Fetch Debug End] ======')
      throw e
    }
  }
}

/**
 * ✅ 强制 baseUrl 生效：改写 origin（保留 pathname+query）
 * - 适用于你自建反代：只要它能按原路径转发即可
 */
function wrapFetchWithBaseUrl(fetchFn: any, baseUrl: string) {
  const base = new URL(normalizeBaseUrl(baseUrl)! + '/')
  return async (input: any, init?: any) => {
    const raw = typeof input === 'string'
      ? input
      : (input?.url ? String(input.url) : String(input))
    const u = new URL(raw)
    u.protocol = base.protocol
    u.host = base.host
    return fetchFn(u.toString(), init)
  }
}

export function createLanguageModel(params: {
  provider: KeyProvider
  apiKey: string
  baseUrl?: string
  model: string
}) {
  const { provider, apiKey, model } = params
  const baseUrl = normalizeBaseUrl(params.baseUrl)

  if (provider === 'google') {
    // google：如果配置了 baseUrl，先强制改写 origin，再套 debug
    const fetchWithBase = baseUrl ? wrapFetchWithBaseUrl(baseFetch, baseUrl) : baseFetch
    const fetchWithDebug = wrapFetchWithDebug(fetchWithBase)

    const google = createGoogleGenerativeAI({
      apiKey,
      ...(baseUrl ? ({ baseURL: baseUrl, baseUrl } as any) : {}),
      fetch: fetchWithDebug,
    } as any)

    return google(model)
  }

  // openai-compatible
  // 注意：这里仍使用 ai-sdk 的 baseURL 逻辑，不强行改写 pathname，避免 baseUrl 带 path 的用户被破坏
  const openai = createOpenAI({
    apiKey,
    ...(baseUrl ? { baseURL: baseUrl } : {}),
    fetch: wrapFetchWithDebug(baseFetch),
  })

  return openai(model)
}
