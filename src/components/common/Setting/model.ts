export class ConfigState {
  timeoutMs?: number
  apiKey?: string
  accessToken?: string
  accessTokenExpiredTime?: string
  apiBaseUrl?: string
  apiModel?: APIMODEL
  reverseProxy?: string
  socksProxy?: string
  socksAuth?: string
  httpsProxy?: string
  balance?: number
  siteConfig?: SiteConfig
  mailConfig?: MailConfig
  auditConfig?: AuditConfig
  announceConfig?: AnnounceConfig
}

export class UserConfig {
  chatModel?: string
}

export class SiteConfig {
  siteTitle?: string
  loginEnabled?: boolean
  loginSalt?: string
  registerEnabled?: boolean
  registerReview?: boolean
  registerMails?: string
  siteDomain?: string
  chatModels?: string
  globalAmount?: number
  usageCountLimit?: boolean
  showWatermark?: boolean
  imageModels?: string
  // ===== 推理模型配置 =====
  reasoningModels?: string
  reasoningEffort?: 'none' | 'low' | 'medium' | 'high' | 'xhigh'
  // ===== ✅ 联网搜索配置 =====
  webSearchEnabled?: boolean
  webSearchProvider?: 'searxng' | 'tavily'
  searxngApiUrl?: string
  tavilyApiKey?: string
  webSearchMaxResults?: number
  webSearchMaxRounds?: number
  webSearchPlannerModel?: string
  hiddenModels?: string

}

export class MailConfig {
  smtpHost?: string
  smtpPort?: number
  smtpTsl?: boolean
  smtpUserName?: string
  smtpPassword?: string
  smtpFrom?: string
}
export type TextAuditServiceProvider = 'baidu'

export interface TextAuditServiceOptions {
  apiKey: string
  apiSecret: string
  label?: string
}
export enum TextAudioType {
  None = 0,
  Request = 1 << 0,
  Response = 1 << 1,
  All = Request | Response,
}

export class AuditConfig {
  enabled?: boolean
  provider?: TextAuditServiceProvider
  options?: TextAuditServiceOptions
  textType?: TextAudioType
  customizeEnabled?: boolean
  sensitiveWords?: string
}

export class AnnounceConfig {
  enabled?: boolean
  announceWords?: string
}

export enum Status {
  Normal = 0,
  Deleted = 1,
  InversionDeleted = 2,
  ResponseDeleted = 3,
  PreVerify = 4,
  AdminVerify = 5,
  Disabled = 6,
}

export enum UserRole {
  Admin = 0,
  User = 1,
  Guest = 2,
  Support = 3,
  Viewer = 4,
  Contributor = 5,
  Developer = 6,
  Tester = 7,
  Partner = 8,
}

export class KeyConfig {
  _id?: string
  key: string
  keyModel: APIMODEL
  chatModels: string[]
  userRoles: UserRole[]
  status: Status
  remark: string
  baseUrl?: string
  constructor(key: string, keyModel: APIMODEL, chatModels: string[], userRoles: UserRole[], remark: string) {
    this.key = key
    this.keyModel = keyModel
    this.chatModels = chatModels
    this.userRoles = userRoles
    this.status = Status.Normal
    this.remark = remark
  }
}

export class UserPrompt {
  _id?: string
  title: string
  value: string
  constructor(title: string, value: string) {
    this.title = title
    this.value = value
  }
}

export type APIMODEL = 'ChatGPTAPI' | 'ChatGPTUnofficialProxyAPI' | undefined

export const apiModelOptions = ['ChatGPTAPI', 'ChatGPTUnofficialProxyAPI'].map((model: string) => {
  return { label: model, key: model, value: model }
})

export const userRoleOptions = Object.values(UserRole).filter(d => Number.isNaN(Number(d))).map((role) => {
  return { label: role as string, key: role as string, value: UserRole[role as keyof typeof UserRole] }
})

export class UserInfo {
  _id?: string
  email?: string
  password?: string
  roles: UserRole[]
  remark?: string
  useAmount?: number
  limit_switch?: boolean
  constructor(roles: UserRole[]) {
    this.roles = roles
  }
}

export class UserPassword {
  oldPassword?: string
  newPassword?: string
  confirmPassword?: string
}

export class TwoFAConfig {
  enaled: boolean
  userName: string
  secretKey: string
  otpauthUrl: string
  testCode: string
  constructor() {
    this.enaled = false
    this.userName = ''
    this.secretKey = ''
    this.otpauthUrl = ''
    this.testCode = ''
  }
}

export interface GiftCard {
  cardno: string
  amount: number
  redeemed: number
}
