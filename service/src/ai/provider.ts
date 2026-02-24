// service/src/ai/provider.ts
import { createOpenAI } from '@ai-sdk/openai'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import nodeFetch from 'node-fetch'

const API_DEBUG = true
const API_DEBUG_FETCH = true

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
    if (headers?.forEach) {
      headers.forEach((v: any, k: any) => { out[String(k)] = v })
    }
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

      console.log('[Response]', {
        status: resp.status,
        statusText: resp.statusText,
        ms,
        contentType: ct,
        headers: respHeaders,
        isStream: ct.includes('text/event-stream')
      })
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
    const fetchWithBase = baseUrl ? wrapFetchWithBaseUrl(baseFetch, baseUrl) : baseFetch
    const fetchWithDebug = wrapFetchWithDebug(fetchWithBase)

    const google = createGoogleGenerativeAI({
      apiKey,
      ...(baseUrl ? ({ baseURL: baseUrl, baseUrl } as any) : {}),
      fetch: fetchWithDebug,
    } as any)

    return google(model)
  }

  const openai = createOpenAI({
    apiKey,
    ...(baseUrl ? { baseURL: baseUrl } : {}),
    fetch: wrapFetchWithDebug(baseFetch),
  })

  return openai(model)
}
