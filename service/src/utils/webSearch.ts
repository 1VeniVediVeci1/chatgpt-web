import fetch from 'node-fetch'

export type WebSearchProvider = 'searxng' | 'tavily'

export interface WebSearchOptions {
  maxResults?: number
  searchDepth?: 'basic' | 'advanced'
  includeDomains?: string[]
  excludeDomains?: string[]
  signal?: AbortSignal
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

function normalizeMaxResults(maxResults?: number): number | undefined {
  if (maxResults == null) return undefined
  const n = Number(maxResults)
  if (!Number.isFinite(n) || n <= 0) return undefined
  return n
}

async function searxngSearch(query: string, apiUrl: string, opts: WebSearchOptions): Promise<WebSearchResults> {
  if (!apiUrl) throw new Error('SearXNG API URL is not configured.')

  const base = apiUrl.replace(/\/+$/, '')
  const url = new URL(`${base}/search`)
  url.searchParams.set('q', query)
  url.searchParams.set('format', 'json')
  url.searchParams.set('categories', 'general')

  if (opts.includeDomains?.length)
    url.searchParams.set('site', opts.includeDomains.join(','))

  console.log(`[WebSearch][SearXNG] Fetching: ${url.toString()}`)

  const fetchOpts: any = {
    method: 'GET',
    headers: { Accept: 'application/json' },
  }

  // ✅ 只在 signal 存在且未 aborted 时传入（避免 node-fetch 兼容性问题）
  if (opts.signal && !opts.signal.aborted) {
    try {
      fetchOpts.signal = opts.signal
    }
    catch {
      // 某些 node-fetch 版本不支持 AbortSignal，忽略
    }
  }

  const resp = await fetch(url.toString(), fetchOpts)

  if (!resp.ok) {
    const body = await resp.text().catch(() => '')
    throw new Error(`[searxng] HTTP ${resp.status} ${resp.statusText}: ${body.slice(0, 200)}`)
  }

  const data = await resp.json() as any
  const rawResults: any[] = Array.isArray(data?.results) ? data.results : []

  console.log(`[WebSearch][SearXNG] Got ${rawResults.length} raw results for query: "${query}"`)

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

  const max = normalizeMaxResults(opts.maxResults)
  if (max != null) results = results.slice(0, max)

  return {
    query: data?.query || query,
    results,
    images: [],
    number_of_results: Number(data?.number_of_results ?? results.length),
  }
}

async function tavilySearch(query: string, apiKey: string, opts: WebSearchOptions): Promise<WebSearchResults> {
  const max = normalizeMaxResults(opts.maxResults)
  const filledQuery = query.length < 5 ? query + ' '.repeat(5 - query.length) : query

  const body: any = {
    api_key: apiKey,
    query: filledQuery,
    search_depth: opts.searchDepth ?? 'advanced',
    include_images: false,
    include_answers: false,
    include_raw_content: true,
    include_domains: opts.includeDomains ?? [],
    exclude_domains: opts.excludeDomains ?? [],
  }

  if (max != null) body.max_results = max

  console.log(`[WebSearch][Tavily] Searching: "${filledQuery}", max_results: ${max ?? 'default'}`)

  const fetchOpts: any = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }

  if (opts.signal && !opts.signal.aborted) {
    try {
      fetchOpts.signal = opts.signal
    }
    catch {
      // 忽略
    }
  }

  const resp = await fetch('https://api.tavily.com/search', fetchOpts)

  if (!resp.ok) {
    const errBody = await resp.text().catch(() => '')
    throw new Error(`[tavily] HTTP ${resp.status} ${resp.statusText}: ${errBody.slice(0, 300)}`)
  }

  const data = await resp.json() as any
  const results: WebSearchResultItem[] = (data?.results || []).map((r: any) => {
    const rawContent = String(r.raw_content || '').trim()
    const snippet = String(r.content || '').trim()
    const bestContent = rawContent.length > snippet.length ? rawContent : snippet
    return {
      title: String(r.title || ''),
      url: String(r.url || ''),
      content: bestContent,
    }
  })

  const finalResults = (max != null) ? results.slice(0, max) : results

  console.log(`[WebSearch][Tavily] Got ${finalResults.length} results for query: "${filledQuery}"`)

  return {
    query,
    results: finalResults,
    images: [],
    number_of_results: finalResults.length,
  }
}

export async function webSearch(query: string, opts: WebSearchOptions = {}): Promise<WebSearchResults> {
  const provider = (opts.provider ?? (process.env.WEB_SEARCH_PROVIDER || process.env.SEARCH_API || 'searxng'))
    .toLowerCase() as WebSearchProvider

  const envMax = normalizeMaxResults(Number(process.env.WEB_SEARCH_MAX_RESULTS))
  const resolvedMaxResults = normalizeMaxResults(opts.maxResults) ?? envMax

  const finalOpts: WebSearchOptions = {
    ...opts,
    ...(resolvedMaxResults != null ? { maxResults: resolvedMaxResults } : {}),
  }

  const k = cacheKey(provider, query, finalOpts)
  const cached = getCached(k)
  if (cached) {
    console.log(`[WebSearch] Cache hit for: "${query}"`)
    return cached
  }

  console.log(`[WebSearch] Provider: ${provider}, Query: "${query}", MaxResults: ${resolvedMaxResults ?? 'default'}`)

  let data: WebSearchResults
  try {
    if (provider === 'tavily') {
      const key = finalOpts.tavilyApiKey || process.env.TAVILY_API_KEY || ''
      if (!key) throw new Error('Tavily API Key is not configured (neither in SiteConfig nor env TAVILY_API_KEY).')
      data = await tavilySearch(query, key, finalOpts)
    }
    else {
      const apiUrl = finalOpts.searxngApiUrl ?? process.env.SEARXNG_API_URL ?? ''
      data = await searxngSearch(query, apiUrl, finalOpts)
    }
  }
  catch (e: any) {
    console.error(`[WebSearch] ERROR for query "${query}":`, e?.message ?? e)
    throw e
  }

  if (data.results.length === 0) {
    console.warn(`[WebSearch] WARNING: 0 results for query "${query}" (provider: ${provider})`)
  }

  setCached(k, data)
  return data
}
