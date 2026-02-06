import fetch from 'node-fetch'

export type WebSearchProvider = 'searxng' | 'tavily'

export interface WebSearchOptions {
  maxResults?: number
  searchDepth?: 'basic' | 'advanced'
  includeDomains?: string[]
  excludeDomains?: string[]
  signal?: AbortSignal
  /** 来自 SiteConfig 的 override */
  provider?: WebSearchProvider
  searxngApiUrl?: string
  tavilyApiKey?: string
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
    opts.searxngApiUrl ?? '',
  ].join('|')
}

function getCached(k: string) {
  const v = memCache.get(k)
  if (!v) return null
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
    throw new Error('SearXNG API URL is not configured.')

  const base = apiUrl.replace(/\/+$/, '')
  const url = new URL(`${base}/search`)
  url.searchParams.set('q', query)
  url.searchParams.set('format', 'json')
  url.searchParams.set('categories', 'general')

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

  if (opts.excludeDomains?.length) {
    results = results.filter(r => {
      try {
        const host = new URL(r.url).hostname
        return !opts.excludeDomains!.some(d => host.includes(d))
      }
      catch { return true }
    })
  }

  const max = clamp(Number(opts.maxResults ?? 5), 1, 20)
  results = results.slice(0, max).map(r => ({ ...r, content: truncText(r.content, 700) }))

  return {
    query: data?.query || query,
    results,
    images: [],
    number_of_results: Number(data?.number_of_results ?? results.length),
  }
}

async function tavilySearch(query: string, apiKey: string, opts: WebSearchOptions): Promise<WebSearchResults> {
  const max = clamp(Number(opts.maxResults ?? 5), 1, 10)
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

  return { query, results: results.slice(0, max), images: [], number_of_results: results.length }
}

/**
 * 执行一次联网搜索：
 * - provider 优先取 opts.provider（SiteConfig），fallback 到环境变量
 * - searxngApiUrl 优先取 opts.searxngApiUrl（SiteConfig），fallback 到环境变量
 * - tavilyApiKey 优先取 opts.tavilyApiKey（SiteConfig），fallback 到环境变量
 */
export async function webSearch(query: string, opts: WebSearchOptions = {}): Promise<WebSearchResults> {
  const provider = (opts.provider ?? (process.env.WEB_SEARCH_PROVIDER || process.env.SEARCH_API || 'searxng')).toLowerCase() as WebSearchProvider
  const maxResults = opts.maxResults ?? Number(process.env.WEB_SEARCH_MAX_RESULTS ?? 5)
  const finalOpts: WebSearchOptions = { ...opts, maxResults }

  const k = cacheKey(provider, query, finalOpts)
  const cached = getCached(k)
  if (cached) return cached

  let data: WebSearchResults
  if (provider === 'tavily') {
    const key = finalOpts.tavilyApiKey || process.env.TAVILY_API_KEY || ''
    if (!key) throw new Error('Tavily API Key is not configured (neither in SiteConfig nor env TAVILY_API_KEY).')
    data = await tavilySearch(query, key, finalOpts)
  }
  else {
    const apiUrl = finalOpts.searxngApiUrl ?? process.env.SEARXNG_API_URL ?? ''
    data = await searxngSearch(query, apiUrl, finalOpts)
  }

  setCached(k, data)
  return data
}
