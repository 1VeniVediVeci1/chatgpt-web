import type OpenAI from 'openai'
import type { ChatCompletionChunk, ChatCompletionRole } from 'openai/resources'
import type { ChatRoom, UsageResponse, UserInfo } from 'src/storage/model'

export interface ChatMessage {
  id: string
  text: MessageContent
  role: ChatCompletionRole
  name?: string
  delta?: string
  // detail?: openai.CreateChatCompletionResponse | CreateChatCompletionStreamResponse
  detail?: ChatCompletionChunk
  parentMessageId?: string
  conversationId?: string
}

export interface RequestOptions {
  message: string
  /**
   * ✅ 是否开启联网搜索（前端开关传入）
   */
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
export type MessageContent = string | Array<OpenAI.ChatCompletionContentPart>

export type ChatResponse = OpenAI.ChatCompletion & {
  text: string
  detail?: {
    usage?: UsageResponse
  }
  conversationId?: string
}
