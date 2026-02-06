import fetch from 'node-fetch'

export type WebSearchProvider = 'searxng' | 'tavily'

export interface WebSearchOptions {
  maxResults?: number
  searchDepth?: 'basic' | 'advanced'
  includeDomains?: string[]
  excludeDomains?: string[]
  signal?: AbortSignal
}

export interface WebSearchResultItem {
  title: string
  url: string
  content: string
}

export interface WebSearchResults {
  query: string
  results: WebSearchResultItem[]
  images: string[]
  number_of_results: number
}

const CACHE_TTL_MS = Number(process.env.WEB_SEARCH_CACHE_TTL_MS ?? 60 * 60 * 1000)
const memCache = new Map<string, { expiresAt: number; data: WebSearchResults }>()

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n))
}

function truncText(s: string, max = 600) {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim()
  return t.length > max ? `${t.slice(0, max)}...` : t
}

function cacheKey(provider: string, query: string, opts: WebSearchOptions) {
  return [
    provider,
    query,
    opts.maxResults ?? '',
    opts.searchDepth ?? '',
    (opts.includeDomains ?? []).join(','),
    (opts.excludeDomains ?? []).join(','),
  ].join('|')
}

function getCached(k: string) {
  const v = memCache.get(k)
  if (!v)
    return null
  if (Date.now() > v.expiresAt) {
    memCache.delete(k)
    return null
  }
  return v.data
}

function setCached(k: string, data: WebSearchResults) {
  memCache.set(k, { expiresAt: Date.now() + CACHE_TTL_MS, data })
}

async function searxngSearch(query: string, apiUrl: string, opts: WebSearchOptions): Promise<WebSearchResults> {
  if (!apiUrl)
    throw new Error('SEARXNG_API_URL is not configured.')

  const base = apiUrl.replace(/\/+$/, '')
  const url = new URL(`${base}/search`)
  url.searchParams.set('q', query)
  url.searchParams.set('format', 'json')
  url.searchParams.set('categories', 'general')

  // includeDomains：部分 searxng 部署支持 site=domain1,domain2（不保证全部支持）
  if (opts.includeDomains?.length)
    url.searchParams.set('site', opts.includeDomains.join(','))

  const resp = await fetch(url.toString(), {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal: opts.signal as any,
  })

  if (!resp.ok)
    throw new Error(`[searxng] ${resp.status} ${resp.statusText}`)

  const data = await resp.json() as any
  const rawResults: any[] = Array.isArray(data?.results) ? data.results : []

  let results: WebSearchResultItem[] = rawResults
    .filter(r => r && !r.img_src && typeof r.url === 'string')
    .map(r => ({
      title: String(r.title || ''),
      url: String(r.url || ''),
      content: String(r.content || ''),
    }))

  // excludeDomains：后置过滤
  if (opts.excludeDomains?.length) {
    results = results.filter(r => {
      try {
        const host = new URL(r.url).hostname
        return !opts.excludeDomains!.some(d => host.includes(d))
      }
      catch {
        return true
      }
    })
  }

  const max = clamp(Number(opts.maxResults ?? 5), 1, 20)
  results = results
    .slice(0, max)
    .map(r => ({ ...r, content: truncText(r.content, 700) }))

  return {
    query: data?.query || query,
    results,
    images: [],
    number_of_results: Number(data?.number_of_results ?? results.length),
  }
}

async function tavilySearch(query: string, apiKey: string, opts: WebSearchOptions): Promise<WebSearchResults> {
  const max = clamp(Number(opts.maxResults ?? 5), 1, 10)

  // Tavily 要求 query 最少 5 字符
  const filledQuery = query.length < 5 ? query + ' '.repeat(5 - query.length) : query

  const resp = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: opts.signal as any,
    body: JSON.stringify({
      api_key: apiKey,
      query: filledQuery,
      max_results: Math.max(max, 5),
      search_depth: opts.searchDepth ?? 'basic',
      include_images: false,
      include_answers: false,
      include_domains: opts.includeDomains ?? [],
      exclude_domains: opts.excludeDomains ?? [],
    }),
  })

  if (!resp.ok)
    throw new Error(`[tavily] ${resp.status} ${resp.statusText}`)

  const data = await resp.json() as any
  const results: WebSearchResultItem[] = (data?.results || []).map((r: any) => ({
    title: String(r.title || ''),
    url: String(r.url || ''),
    content: truncText(String(r.content || ''), 900),
  }))

  return {
    query,
    results: results.slice(0, max),
    images: [],
    number_of_results: results.length,
  }
}

/**
 * 执行一次联网搜索（provider 由环境变量控制）：
 * - WEB_SEARCH_PROVIDER 或 SEARCH_API：searxng / tavily
 * - SEARXNG_API_URL
 * - TAVILY_API_KEY
 * - WEB_SEARCH_MAX_RESULTS
 */
export async function webSearch(query: string, opts: WebSearchOptions = {}): Promise<WebSearchResults> {
  const provider = String(process.env.WEB_SEARCH_PROVIDER || process.env.SEARCH_API || 'searxng').toLowerCase() as WebSearchProvider
  const maxResults = opts.maxResults ?? Number(process.env.WEB_SEARCH_MAX_RESULTS ?? 5)
  const finalOpts: WebSearchOptions = { ...opts, maxResults }

  const k = cacheKey(provider, query, finalOpts)
  const cached = getCached(k)
  if (cached)
    return cached

  let data: WebSearchResults
  if (provider === 'tavily') {
    const key = process.env.TAVILY_API_KEY
    if (!key)
      throw new Error('TAVILY_API_KEY is not configured.')
    data = await tavilySearch(query, key, finalOpts)
  }
  else {
    const apiUrl = process.env.SEARXNG_API_URL || ''
    data = await searxngSearch(query, apiUrl, finalOpts)
  }

  setCached(k, data)
  return data
}

/**
 * （可选）把搜索结果格式化成可直接拼进 prompt 的文本（带编号引用）
 */
export function formatWebSearchResultsForPrompt(r: WebSearchResults): string {
  const lines = (r.results || []).map((it, idx) => {
    const n = idx + 1
    return [
      `[${n}] ${it.title}`.trim(),
      `URL: ${it.url}`,
      `摘要: ${it.content}`.trim(),
    ].filter(Boolean).join('\n')
  })

  return [
    '【联网搜索结果（请优先使用并按编号引用）】',
    ...lines,
    '',
    '【回答要求】',
    '- 结合以上来源回答；涉及事实/数据/结论请在句末用 [编号] 引用（可多个）。',
    '- 若来源不足以支持结论，请明确说明“不确定/需更多信息”。',
  ].join('\n')
}
