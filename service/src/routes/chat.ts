import { ObjectId } from 'mongodb'
import type { TiktokenModel } from 'gpt-token'
import { textTokens } from 'gpt-token'
import Router from 'express'
import type OpenAI from 'openai'
import type { ChatMessage, ChatResponse } from 'src/chatgpt/types'
import { limiter } from '../middleware/limiter'
import { abortChatProcess, chatReplyProcess, containsSensitiveWords, getChatProcessState } from '../chatgpt'
import { auth } from '../middleware/auth'
import { getCacheConfig } from '../storage/config'
import type { ChatInfo, ChatOptions, UserInfo } from '../storage/model'
import { Status, UsageResponse, UserRole } from '../storage/model'
import {
  clearChat,
  deleteAllChatRooms,
  deleteChat,
  existsChatRoom,
  getChat, getChatRoom,
  getChats,
  getUserById, insertChat, insertChatUsage, updateAmountMinusOne, updateChat,
} from '../storage/mongo'
import { isNotEmptyString } from '../utils/is'
import type { RequestProps } from '../types'

export const router = Router()

router.get('/chat-history', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    const roomId = +req.query.roomId
    const lastId = req.query.lastId as string
    const all = req.query.all as string
    if ((!roomId || !await existsChatRoom(userId, roomId)) && (all === null || all === 'undefined' || all === undefined || all.trim().length === 0)) {
      res.send({ status: 'Success', message: null, data: [] })
      return
    }

    if (all !== null && all !== 'undefined' && all !== undefined && all.trim().length !== 0) {
      const config = await getCacheConfig()
      if (config.siteConfig.loginEnabled) {
        try {
          const user = await getUserById(userId)
          if (user == null || user.status !== Status.Normal || !user.roles.includes(UserRole.Admin)) {
            res.send({ status: 'Fail', message: '无权限 | No permission.', data: null })
            return
          }
        }
        catch (error) {
          res.send({ status: 'Unauthorized', message: error.message ?? 'Please authenticate.', data: null })
        }
      }
      else {
        res.send({ status: 'Fail', message: '无权限 | No permission.', data: null })
      }
    }

    const chats = await getChats(roomId, !isNotEmptyString(lastId) ? null : Number.parseInt(lastId), all)
    const result = []
    chats.forEach((c) => {
      if (c.status !== Status.InversionDeleted) {
        result.push({
          uuid: c.uuid,
          model: c.model,
          dateTime: new Date(c.dateTime).toLocaleString(),
          text: c.prompt,
          images: c.images,
          inversion: true,
          error: false,
          conversationOptions: null,
          requestOptions: {
            prompt: c.prompt,
            options: null,
          },
        })
      }
      if (c.status !== Status.ResponseDeleted) {
        const usage = c.options.completion_tokens
          ? {
              completion_tokens: c.options.completion_tokens || null,
              prompt_tokens: c.options.prompt_tokens || null,
              total_tokens: c.options.total_tokens || null,
              estimated: c.options.estimated || null,
            }
          : undefined
        result.push({
          uuid: c.uuid,
          dateTime: new Date(c.dateTime).toLocaleString(),
          text: c.response,
          inversion: false,
          error: false,
          loading: false,
          responseCount: (c.previousResponse?.length ?? 0) + 1,
          conversationOptions: {
            parentMessageId: c.options.messageId,
            conversationId: c.options.conversationId,
          },
          requestOptions: {
            prompt: c.prompt,
            parentMessageId: c.options.parentMessageId,
            options: {
              parentMessageId: c.options.messageId,
              conversationId: c.options.conversationId,
            },
          },
          usage,
        })
      }
    })

    res.send({ status: 'Success', message: null, data: result })
  }
  catch (error) {
    console.error(error)
    res.send({ status: 'Fail', message: 'Load error', data: null })
  }
})

router.get('/chat-response-history', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    const roomId = +req.query.roomId
    const uuid = +req.query.uuid
    const index = +req.query.index
    if (!roomId || !await existsChatRoom(userId, roomId)) {
      res.send({ status: 'Success', message: null, data: [] })
      return
    }
    const chat = await getChat(roomId, uuid)
    if (chat.previousResponse === undefined || chat.previousResponse.length < index) {
      res.send({ status: 'Fail', message: 'Error', data: [] })
      return
    }
    const response = index >= chat.previousResponse.length
      ? chat
      : chat.previousResponse[index]
    const usage = response.options.completion_tokens
      ? {
          completion_tokens: response.options.completion_tokens || null,
          prompt_tokens: response.options.prompt_tokens || null,
          total_tokens: response.options.total_tokens || null,
          estimated: response.options.estimated || null,
        }
      : undefined
    res.send({
      status: 'Success',
      message: null,
      data: {
        uuid: chat.uuid,
        dateTime: new Date(chat.dateTime).toLocaleString(),
        text: response.response,
        inversion: false,
        error: false,
        loading: false,
        responseCount: (chat.previousResponse?.length ?? 0) + 1,
        conversationOptions: {
          parentMessageId: response.options.messageId,
          conversationId: response.options.conversationId,
        },
        requestOptions: {
          prompt: chat.prompt,
          parentMessageId: response.options.parentMessageId,
          options: {
            parentMessageId: response.options.messageId,
            conversationId: response.options.conversationId,
          },
        },
        usage,
      },
    })
  }
  catch (error) {
    console.error(error)
    res.send({ status: 'Fail', message: 'Load error', data: null })
  }
})

router.post('/chat-delete', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    const { roomId, uuid, inversion } = req.body as { roomId: number; uuid: number; inversion: boolean }
    if (!roomId || !await existsChatRoom(userId, roomId)) {
      res.send({ status: 'Fail', message: 'Unknown room', data: null })
      return
    }
    await deleteChat(roomId, uuid, inversion)
    res.send({ status: 'Success', message: null, data: null })
  }
  catch (error) {
    console.error(error)
    res.send({ status: 'Fail', message: 'Delete error', data: null })
  }
})

router.post('/chat-clear-all', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    await deleteAllChatRooms(userId)
    res.send({ status: 'Success', message: null, data: null })
  }
  catch (error) {
    console.error(error)
    res.send({ status: 'Fail', message: 'Delete error', data: null })
  }
})

router.post('/chat-clear', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    const { roomId } = req.body as { roomId: number }
    if (!roomId || !await existsChatRoom(userId, roomId)) {
      res.send({ status: 'Fail', message: 'Unknown room', data: null })
      return
    }
    await clearChat(roomId)
    res.send({ status: 'Success', message: null, data: null })
  }
  catch (error) {
    console.error(error)
    res.send({ status: 'Fail', message: 'Delete error', data: null })
  }
})

/**
 * ✅ Job化：后台执行生成并落库
 * - 不再 res.write stream
 * - 前端通过 /chat-process-status 轮询，结束后再 /chat-history 拉取
 */
async function runChatJobInBackground(params: {
  userId: string
  roomId: number
  uuid: number
  regenerate: boolean
  prompt: string
  uploadFileKeys?: string[]
  options: any
  systemMessage?: string
  temperature?: number
  top_p?: number
  model: string
  room: any
  message: ChatInfo
  user: UserInfo
}) {
  const {
    userId,
    roomId,
    uuid,
    regenerate,
    prompt,
    uploadFileKeys,
    options,
    systemMessage,
    temperature,
    top_p,
    model,
    room,
    message,
    user,
  } = params

  const config = await getCacheConfig()

  let lastResponse: ChatMessage | undefined
  let result: null | {
    message: string
    data: ChatResponse
    status: string
  } = null

  try {
    result = await chatReplyProcess({
      message: prompt,
      uploadFileKeys,
      lastContext: options,
      // 可选：想写“中间结果”到DB可以在这里节流 updateChat
      process: (chat: ChatMessage) => {
        lastResponse = chat
      },
      systemMessage,
      temperature,
      top_p,
      user,
      messageId: message._id.toString(),
      tryCount: 0,
      room,
    })

    // 补 usage（与原逻辑一致）
    if (!result.data.detail?.usage) {
      if (!result.data.detail)
        result.data.detail = {}
      result.data.detail.usage = new UsageResponse()
      result.data.detail.usage.prompt_tokens = textTokens(prompt, 'gpt-3.5-turbo' as TiktokenModel)
      result.data.detail.usage.completion_tokens = textTokens(result.data.text, 'gpt-3.5-turbo' as TiktokenModel)
      result.data.detail.usage.total_tokens = result.data.detail.usage.prompt_tokens + result.data.detail.usage.completion_tokens
      result.data.detail.usage.estimated = true
    }
  }
  catch (error: any) {
    console.error('[Job] chatReplyProcess error:', error)

    // 确保 finally 中能拿到 result
    result = {
      status: 'Fail',
      message: error?.message ?? 'Please check the back-end console',
      data: (lastResponse ?? { text: error?.message ?? '' } as any),
    }
  }
  finally {
    // === 与原 finally 等价的安全兜底 ===
    const safeResult = result ?? {
      status: 'Fail',
      message: lastResponse?.text ?? 'Unknown error',
      data: lastResponse ?? undefined,
    }

    if (!safeResult.data && lastResponse)
      safeResult.data = lastResponse as any

    if (!safeResult.data)
      return

    if (safeResult.status !== 'Success') {
      lastResponse = lastResponse ?? { text: safeResult.message } as any
      safeResult.data = lastResponse as any
    }

    const resultData = safeResult.data as any

    try {
      // 写入主表 chat（与原逻辑一致）
      if (regenerate && message.options?.messageId) {
        const previousResponse = message.previousResponse || []
        previousResponse.push({ response: message.response, options: message.options })
        await updateChat(
          message._id as unknown as string,
          resultData.text,
          resultData.id,
          resultData.conversationId,
          model,
          resultData.detail?.usage as UsageResponse,
          previousResponse as [],
        )
      }
      else {
        await updateChat(
          message._id as unknown as string,
          resultData.text,
          resultData.id,
          resultData.conversationId,
          model,
          resultData.detail?.usage as UsageResponse,
        )
      }

      // usage 记录
      if (resultData.detail?.usage) {
        await insertChatUsage(
          ObjectId.createFromHexString(userId),
          roomId,
          message._id,
          resultData.id,
          model,
          resultData.detail?.usage as UsageResponse,
        )
      }

      // 次数扣减（与原逻辑一致）
      if (config.siteConfig?.usageCountLimit) {
        if (userId !== '6406d8c50aedd633885fa16f' && user && user.useAmount && user.limit_switch)
          await updateAmountMinusOne(userId)
      }
    }
    catch (e) {
      console.error('[Job] persist result failed:', e)
    }
  }
}

router.post('/chat-process', [auth, limiter], async (req, res) => {
  let { roomId, uuid, regenerate, prompt, uploadFileKeys, options = {}, systemMessage, temperature, top_p } = req.body as RequestProps
  const userId = req.headers.userId.toString()
  const config = await getCacheConfig()

  // ✅ 防止一个用户同时开多个后台任务导致资源失控（你也可以改成排队）
  const state = getChatProcessState(userId)
  if (state.isProcessing) {
    res.send({ status: 'Fail', message: '当前已有生成任务在进行中，请稍后或先停止生成。', data: null })
    return
  }

  const room = await getChatRoom(userId, roomId)
  if (room == null)
    globalThis.console.error(`Unable to get chat room \t ${userId}\t ${roomId}`)
  if (room != null && isNotEmptyString(room.prompt))
    systemMessage = room.prompt
  const model = room.chatModel

  let user = await getUserById(userId)

  try {
    // guest
    if (userId === '6406d8c50aedd633885fa16f') {
      user = { _id: userId, roles: [UserRole.User], useAmount: 999, advanced: { maxContextCount: 999 }, limit_switch: false } as any
    }
    else {
      if (config.siteConfig?.usageCountLimit) {
        const useAmount = user ? (user.useAmount ?? 0) : 0
        if (useAmount <= 0 && user.limit_switch) {
          res.send({ status: 'Fail', message: '提问次数用完啦 | Question limit reached', data: null })
          return
        }
      }
    }

    // audit
    if (config.auditConfig.enabled || config.auditConfig.customizeEnabled) {
      if (!user.roles.includes(UserRole.Admin) && await containsSensitiveWords(config.auditConfig, prompt)) {
        res.send({ status: 'Fail', message: '含有敏感词 | Contains sensitive words', data: null })
        return
      }
    }

    // 创建/获取 chat 记录（与原逻辑一致）
    const message: ChatInfo = regenerate
      ? await getChat(roomId, uuid)
      : await insertChat(uuid, prompt, uploadFileKeys, roomId, model, options as ChatOptions)

    // ✅ 立刻返回“已受理”，让前端去轮询
    res.send({
      status: 'Success',
      message: 'Accepted',
      data: {
        jobId: message._id.toString(),
        roomId,
        uuid,
      },
    })

    // ✅ 后台异步执行（不 await）
    void runChatJobInBackground({
      userId,
      roomId,
      uuid,
      regenerate,
      prompt,
      uploadFileKeys,
      options,
      systemMessage,
      temperature,
      top_p,
      model,
      room,
      message,
      user,
    })
  }
  catch (error: any) {
    console.error(error)
    res.send({ status: 'Fail', message: error?.message ?? 'Submit job failed', data: null })
  }
})

router.post('/chat-abort', [auth, limiter], async (req, res) => {
  try {
    const userId = req.headers.userId.toString()
    const { text, messageId, conversationId } = req.body as { text: string; messageId: string; conversationId: string }
    const msgId = await abortChatProcess(userId)
    await updateChat(msgId,
      text,
      messageId,
      conversationId,
      null, null)
    res.send({ status: 'Success', message: 'OK', data: null })
  }
  catch (error) {
    res.send({ status: 'Fail', message: 'Abort failed', data: null })
  }
})

router.post('/chat-process-status', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    const status = getChatProcessState(userId)
    res.send({ status: 'Success', message: '', data: status })
  }
  catch (error) {
    res.send({ status: 'Fail', message: error.message, data: null })
  }
})
