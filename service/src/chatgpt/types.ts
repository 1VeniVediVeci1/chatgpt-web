import type OpenAI from 'openai'
import type { ChatCompletionChunk, ChatCompletionRole } from 'openai/resources'
import type { ChatRoom, UserInfo } from 'src/storage/model'

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
