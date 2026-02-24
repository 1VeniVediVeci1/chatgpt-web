// service/src/ai/provider.ts
import { createOpenAI } from '@ai-sdk/openai'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import nodeFetch from 'node-fetch'

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
    const google = createGoogleGenerativeAI({
      apiKey,
      ...(baseUrl ? ({ baseURL: baseUrl, baseUrl } as any) : {}),
      ...(baseUrl ? ({ fetch: wrapFetchWithBaseUrl(baseFetch, baseUrl) } as any) : {}),
    } as any)
    return google(model)
  }

  // openai-compatible
  const openai = createOpenAI({
    apiKey,
    ...(baseUrl ? { baseURL: baseUrl } : {}),
  })
  return openai(model)
}
