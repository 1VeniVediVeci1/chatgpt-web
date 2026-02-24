import { ObjectId } from 'mongodb'
import * as dotenv from 'dotenv'
import type { TextAuditServiceProvider } from 'src/utils/textAudit'
import { isNotEmptyString, isTextAuditServiceProvider } from '../utils/is'
import { AdvancedConfig, AnnounceConfig, AuditConfig, Config, KeyConfig, MailConfig, SiteConfig, TextAudioType, UserRole } from './model'
import { getConfig, getKeys, upsertKey } from './mongo'

dotenv.config()

let cachedConfig: Config | undefined
let cacheExpiration = 0

export async function getCacheConfig(): Promise<Config> {
  const now = Date.now()
  if (cachedConfig && cacheExpiration > now)
    return Promise.resolve(cachedConfig)

  const loadedConfig = await getOriginConfig()

  cachedConfig = loadedConfig
  cacheExpiration = now + 10 * 60 * 1000

  return Promise.resolve(cachedConfig)
}

function normalizeApiModel(input: any): 'openai-compatible' | 'google' {
  const v = String(input ?? '').trim()
  if (v === 'google') return 'google'
  return 'openai-compatible'
}

export async function getOriginConfig() {
  let config = await getConfig()
  if (config == null) {
    config = new Config(
      new ObjectId(),
      !Number.isNaN(+process.env.TIMEOUT_MS) ? +process.env.TIMEOUT_MS : 600 * 1000,
      process.env.OPENAI_API_KEY,
      process.env.OPENAI_API_DISABLE_DEBUG === 'true',
      process.env.OPENAI_ACCESS_TOKEN, // legacy（不再使用）
      process.env.OPENAI_API_BASE_URL,
      normalizeApiModel(process.env.OPENAI_API_MODEL),
      process.env.API_REVERSE_PROXY, // legacy（不再使用）
      (process.env.SOCKS_PROXY_HOST && process.env.SOCKS_PROXY_PORT)
        ? (`${process.env.SOCKS_PROXY_HOST}:${process.env.SOCKS_PROXY_PORT}`)
        : '',
      (process.env.SOCKS_PROXY_USERNAME && process.env.SOCKS_PROXY_PASSWORD)
        ? (`${process.env.SOCKS_PROXY_USERNAME}:${process.env.SOCKS_PROXY_PASSWORD}`)
        : '',
      process.env.HTTPS_PROXY,
      new SiteConfig(
        process.env.SITE_TITLE || 'ChatGPT Web',
        isNotEmptyString(process.env.AUTH_SECRET_KEY),
        process.env.AUTH_PROXY_ENABLED === 'true',
        process.env.AUTH_SECRET_KEY,
        process.env.REGISTER_ENABLED === 'true',
        process.env.REGISTER_REVIEW === 'true',
        process.env.REGISTER_MAILS,
        process.env.SITE_DOMAIN,
      ),
      new MailConfig(
        process.env.SMTP_HOST,
        !Number.isNaN(+process.env.SMTP_PORT) ? +process.env.SMTP_PORT : 465,
        process.env.SMTP_TSL === 'true',
        process.env.SMTP_USERNAME,
        process.env.SMTP_PASSWORD,
        process.env.SMTP_FROM || process.env.SMTP_USERNAME,
      ),
    )
  }
  else {
    if (config.siteConfig.loginEnabled === undefined)
      config.siteConfig.loginEnabled = isNotEmptyString(process.env.AUTH_SECRET_KEY)
    if (config.siteConfig.authProxyEnabled === undefined)
      config.siteConfig.authProxyEnabled = process.env.AUTH_PROXY_ENABLED === 'true'
    if (config.siteConfig.loginSalt === undefined)
      config.siteConfig.loginSalt = process.env.AUTH_SECRET_KEY
    if (config.apiDisableDebug === undefined)
      config.apiDisableDebug = process.env.OPENAI_API_DISABLE_DEBUG === 'true'
    if (config.socksAuth === undefined) {
      config.socksAuth = (process.env.SOCKS_PROXY_USERNAME && process.env.SOCKS_PROXY_PASSWORD)
        ? (`${process.env.SOCKS_PROXY_USERNAME}:${process.env.SOCKS_PROXY_PASSWORD}`)
        : ''
    }
    if (config.siteConfig.registerReview === undefined)
      config.siteConfig.registerReview = process.env.REGISTER_REVIEW === 'true'
    if (!config.apiModel)
      config.apiModel = 'openai-compatible'
  }

  if (config.auditConfig === undefined) {
    config.auditConfig = new AuditConfig(
      process.env.AUDIT_ENABLED === 'true',
      isTextAuditServiceProvider(process.env.AUDIT_PROVIDER)
        ? process.env.AUDIT_PROVIDER as TextAuditServiceProvider
        : 'baidu',
      {
        apiKey: process.env.AUDIT_API_KEY,
        apiSecret: process.env.AUDIT_API_SECRET,
        label: process.env.AUDIT_TEXT_LABEL,
      },
      getTextAuditServiceOptionFromString(process.env.AUDIT_TEXT_TYPE),
      false,
      '',
    )
  }

  if (!config.advancedConfig) {
    config.advancedConfig = new AdvancedConfig(
      'You are ChatGPT, a large language model trained by OpenAI. Follow the user\'s instructions carefully.Respond using markdown (latex start with $).',
      0.8,
      1,
      20,
    )
  }

  if (!config.announceConfig) {
    config.announceConfig = new AnnounceConfig(
      false,
      '',
    )
  }

  // 默认值处理
  if (!isNotEmptyString(config.siteConfig.chatModels))
    config.siteConfig.chatModels = 'gpt-3.5-turbo,gpt-4-turbo-preview,gpt-4-vision-preview'

  // ===== 推理默认值 =====
  if (config.siteConfig.reasoningModels === undefined)
    config.siteConfig.reasoningModels = ''
  if (config.siteConfig.reasoningEffort === undefined)
    config.siteConfig.reasoningEffort = 'medium'

  // ===== ✅ 联网搜索默认值 =====
  if (config.siteConfig.webSearchEnabled === undefined)
    config.siteConfig.webSearchEnabled = false
  if (config.siteConfig.webSearchProvider === undefined)
    config.siteConfig.webSearchProvider = (process.env.WEB_SEARCH_PROVIDER as any) || (process.env.SEARCH_API as any) || 'searxng'
  if (config.siteConfig.searxngApiUrl === undefined)
    config.siteConfig.searxngApiUrl = process.env.SEARXNG_API_URL || ''
  if (config.siteConfig.tavilyApiKey === undefined)
    config.siteConfig.tavilyApiKey = process.env.TAVILY_API_KEY || ''
  if (config.siteConfig.webSearchMaxResults === undefined)
    config.siteConfig.webSearchMaxResults = Number(process.env.WEB_SEARCH_MAX_RESULTS || 5)
  if (config.siteConfig.webSearchMaxRounds === undefined)
    config.siteConfig.webSearchMaxRounds = Number(process.env.WEB_SEARCH_MAX_ROUNDS || 3)
  if (config.siteConfig.webSearchPlannerModel === undefined)
    config.siteConfig.webSearchPlannerModel = process.env.WEB_SEARCH_PLANNER_MODEL || ''
  if (config.siteConfig.hiddenModels === undefined)
    config.siteConfig.hiddenModels = ''

  return config
}

function getTextAuditServiceOptionFromString(value: string): TextAudioType {
  if (value === undefined)
    return TextAudioType.None

  switch (value.toLowerCase()) {
    case 'request':
      return TextAudioType.Request
    case 'response':
      return TextAudioType.Response
    case 'all':
      return TextAudioType.All
    default:
      return TextAudioType.None
  }
}

export function clearConfigCache() {
  cacheExpiration = 0
  cachedConfig = null
}

let apiKeysCachedConfig: KeyConfig[] | undefined
let apiKeysCacheExpiration = 0

export async function getCacheApiKeys(): Promise<KeyConfig[]> {
  const now = Date.now()
  if (apiKeysCachedConfig && apiKeysCacheExpiration > now)
    return Promise.resolve(apiKeysCachedConfig)

  const loadedConfig = (await getApiKeys()).keys

  apiKeysCachedConfig = loadedConfig
  apiKeysCacheExpiration = now + 10 * 60 * 1000

  return Promise.resolve(apiKeysCachedConfig)
}

export function clearApiKeyCache() {
  apiKeysCacheExpiration = 0
  getCacheApiKeys()
}

export async function getApiKeys() {
  const result = await getKeys()
  const config = await getCacheConfig()

  // ✅ 没有 key 时：用全局 apiKey 自动补一条 openai-compatible（兼容旧行为）
  if (result.keys.length <= 0) {
    if (isNotEmptyString(config.apiKey))
      result.keys.push(await upsertKey(new KeyConfig(config.apiKey, 'openai-compatible', [], [], '')))
    result.total = result.keys.length
  }

  result.keys.forEach((key) => {
    if (key.userRoles == null || key.userRoles.length <= 0) {
      key.userRoles.push(UserRole.Admin)
      key.userRoles.push(UserRole.User)
      key.userRoles.push(UserRole.Guest)
    }
    if (key.chatModels == null || key.chatModels.length <= 0) {
      config.siteConfig.chatModels.split(',').forEach((chatModel) => {
        key.chatModels.push(chatModel)
      })
    }
  })
  return result
}

export const authProxyHeaderName = process.env.AUTH_PROXY_HEADER_NAME ?? 'X-Email'
