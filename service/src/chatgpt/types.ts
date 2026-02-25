// service/src/chatgpt/types.ts
import type { ChatRoom, UsageResponse, UserInfo } from 'src/storage/model'

export type ChatRole = 'system' | 'user' | 'assistant'

export type MessagePart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }
  | { type: 'file'; mimeType: string; data: string }

export type MessageContent = string | MessagePart[]

export interface ChatMessage {
  id: string
  text: MessageContent
  role: ChatRole
  name?: string
  delta?: string
  detail?: any
  parentMessageId?: string
  conversationId?: string
}

export interface RequestOptions {
  message: string
  searchMode?: boolean
  uploadFileKeys?: string[]
  lastContext?: { conversationId?: string; parentMessageId?: string }
  process?: (chat: ChatMessage) => void
  systemMessage?: string
  temperature?: number
  top_p?: number
  user: UserInfo
  messageId: string
  tryCount: number
  room: ChatRoom
}

export interface BalanceResponse {
  total_usage: number
}

export type ChatResponse = {
  text: string
  detail?: {
    usage?: UsageResponse
  }
  conversationId?: string
  object?: string
  choices?: any[]
  created?: number
  model?: string
  id?: string
}
