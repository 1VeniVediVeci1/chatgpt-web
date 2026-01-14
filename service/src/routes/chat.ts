import { ObjectId } from 'mongodb'
import type { TiktokenModel } from 'gpt-token'
import { textTokens } from 'gpt-token'
import Router from 'express'
import { limiter } from '../middleware/limiter'
import { abortChatProcess, chatReplyProcess, containsSensitiveWords } from '../chatgpt'
import { auth } from '../middleware/auth'
import { getCacheConfig } from '../storage/config'
import type { ChatInfo, ChatOptions, UserInfo } from '../storage/model'
import { Status, UsageResponse, UserRole } from '../storage/model'
import {
  clearChat,
  deleteAllChatRooms,
  deleteChat,
  existsChatRoom,
  getChat,
  getChatRoom,
  getChats,
  getUserById,
  insertChat,
  insertChatUsage,
  updateAmountMinusOne,
  updateChat,
  updateChatResponsePartial,
} from '../storage/mongo'
import { isNotEmptyString } from '../utils/is'
import type { RequestProps } from '../types'

export const router = Router()

/**
 * ✅ Job 状态（按 userId+roomId 管理）
 */
type JobState = {
  userId: string
  roomId: number
  uuid: number
  chatId: string
  status: 'running' | 'done'
  updatedAt: number
}
const jobStates = new Map<string, JobState>()
const jobKey = (userId: string, roomId: number) => `${userId}:${roomId}`

function countRunningJobs(userId: string) {
  let n = 0
  for (const v of jobStates.values()) {
    if (v.userId === userId && v.status === 'running') n++
  }
  return n
}

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
        catch (error: any) {
          res.send({ status: 'Unauthorized', message: error.message ?? 'Please authenticate.', data: null })
        }
      }
      else {
        res.send({ status: 'Fail', message: '无权限 | No permission.', data: null })
      }
    }

    const chats = await getChats(roomId, !isNotEmptyString(lastId) ? null : Number.parseInt(lastId), all)
    const result: any[] = []

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
        const usage = c.options?.completion_tokens
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
          conversationOptions: c.options?.messageId
            ? {
                parentMessageId: c.options.messageId,
                conversationId: c.options.conversationId,
              }
            : null,
          requestOptions: {
            prompt: c.prompt,
            parentMessageId: c.options?.parentMessageId,
            options: c.options?.messageId
              ? {
                  parentMessageId: c.options.messageId,
                  conversationId: c.options.conversationId,
                }
              : null,
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
    const usage = response.options?.completion_tokens
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

/**
 * ✅ 伪流式：返回某条 uuid 的最新 response
 */
router.get('/chat-latest', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    const roomId = +req.query.roomId
    const uuid = +req.query.uuid

    if (!roomId || !await existsChatRoom(userId, roomId)) {
      res.send({ status: 'Fail', message: 'Unknown room', data: null })
      return
    }

    const chat = await getChat(roomId, uuid)
    if (!chat) {
      res.send({ status: 'Fail', message: 'Chat not found', data: null })
      return
    }

    const key = jobKey(userId, roomId)
    const job = jobStates.get(key)
    const isProcessing = !!(job?.status === 'running' && job.uuid === uuid)

    const usage = chat.options?.completion_tokens
      ? {
          completion_tokens: chat.options.completion_tokens || null,
          prompt_tokens: chat.options.prompt_tokens || null,
          total_tokens: chat.options.total_tokens || null,
          estimated: chat.options.estimated || null,
        }
      : undefined

    const conversationOptions = chat.options?.messageId
      ? {
          parentMessageId: chat.options.messageId,
          conversationId: chat.options.conversationId,
        }
      : null

    res.send({
      status: 'Success',
      message: null,
      data: {
        text: chat.response || '',
        isProcessing,
        conversationOptions,
        usage,
        chatId: chat._id?.toString(),
      },
    })
  }
  catch (e: any) {
    res.send({ status: 'Fail', message: e?.message ?? 'Error', data: null })
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

  let lastResponse: any
  let result: any = null

  const FLUSH_INTERVAL_MS = Number(process.env.JOB_FLUSH_MS ?? 800)
  let lastFlushTime = 0
  let lastFlushedLen = 0

  try { await updateChatResponsePartial(message._id.toString(), '') } catch {}

  try {
    result = await chatReplyProcess({
      message: prompt,
      uploadFileKeys,
      lastContext: options,
      process: (chat: any) => {
        lastResponse = chat
        if (typeof chat?.text !== 'string') return

        const now = Date.now()
        const len = chat.text.length
        if (now - lastFlushTime < FLUSH_INTERVAL_MS) return
        if (len <= lastFlushedLen) return

        lastFlushTime = now
        lastFlushedLen = len

        void updateChatResponsePartial(message._id.toString(), chat.text).catch(() => {})
      },
      systemMessage,
      temperature,
      top_p,
      user,
      messageId: message._id.toString(),
      tryCount: 0,
      room,
    })

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
    const isAbort =
      String(error?.name).includes('Abort')
      || String(error?.message).toLowerCase().includes('aborted')
      || String(error?.message).toLowerCase().includes('canceled')

    console.error('[Job] chatReplyProcess error:', error?.message ?? error)

    if (isAbort && lastResponse?.text) {
      result = {
        status: 'Success',
        data: {
          text: lastResponse.text,
          id: lastResponse.id,
          conversationId: lastResponse.conversationId,
          detail: lastResponse.detail,
        },
      }
    }
    else {
      result = {
        status: 'Fail',
        message: error?.message ?? 'Please check the back-end console',
        data: (lastResponse ?? { text: error?.message ?? '' }),
      }
    }
  }
  finally {
    const key = jobKey(userId, roomId)

    const safeResult = result ?? {
      status: 'Fail',
      message: lastResponse?.text ?? 'Unknown error',
      data: lastResponse ?? undefined,
    }

    if (!safeResult.data && lastResponse)
      safeResult.data = lastResponse

    if (!safeResult.data) {
      jobStates.set(key, { userId, roomId, uuid, chatId: message._id.toString(), status: 'done', updatedAt: Date.now() })
      return
    }

    if (safeResult.status !== 'Success') {
      lastResponse = lastResponse ?? { text: safeResult.message }
      safeResult.data = lastResponse
    }

    const resultData = safeResult.data as any

    try {
      if (typeof resultData.text === 'string')
        await updateChatResponsePartial(message._id.toString(), resultData.text)
    } catch {}

    try {
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

      if (config.siteConfig?.usageCountLimit) {
        if (userId !== '6406d8c50aedd633885fa16f' && user && user.useAmount && user.limit_switch)
          await updateAmountMinusOne(userId)
      }
    }
    catch (e) {
      console.error('[Job] persist result failed:', e)
    }
    finally {
      jobStates.set(key, {
        userId,
        roomId,
        uuid,
        chatId: message._id.toString(),
        status: 'done',
        updatedAt: Date.now(),
      })

      setTimeout(() => {
        const s = jobStates.get(key)
        if (s?.status === 'done' && Date.now() - s.updatedAt > 55_000)
          jobStates.delete(key)
      }, 60_000)
    }
  }
}

router.post('/chat-process', [auth, limiter], async (req, res) => {
  let { roomId, uuid, regenerate, prompt, uploadFileKeys, options = {}, systemMessage, temperature, top_p } = req.body as RequestProps
  const userId = req.headers.userId.toString()
  const config = await getCacheConfig()

  const key = jobKey(userId, roomId)
  const existing = jobStates.get(key)
  if (existing?.status === 'running') {
    res.send({ status: 'Fail', message: '该会话正在生成中，请稍后或先停止生成。', data: null })
    return
  }

  // ✅ 限制每个用户最多 3 个并发（跨房间）
  const maxJobs = Number(process.env.MAX_RUNNING_JOBS_PER_USER ?? 3)
  if (countRunningJobs(userId) >= maxJobs) {
    res.send({ status: 'Fail', message: `并发生成达到上限(${maxJobs})，请先等待已有任务完成或停止。`, data: null })
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

    if (config.auditConfig.enabled || config.auditConfig.customizeEnabled) {
      if (!user.roles.includes(UserRole.Admin) && await containsSensitiveWords(config.auditConfig, prompt)) {
        res.send({ status: 'Fail', message: '含有敏感词 | Contains sensitive words', data: null })
        return
      }
    }

    const message: ChatInfo = regenerate
      ? await getChat(roomId, uuid)
      : await insertChat(uuid, prompt, uploadFileKeys, roomId, model, options as ChatOptions)

    jobStates.set(key, {
      userId,
      roomId,
      uuid,
      chatId: message._id.toString(),
      status: 'running',
      updatedAt: Date.now(),
    })

    res.send({
      status: 'Success',
      message: 'Accepted',
      data: { jobId: message._id.toString(), roomId, uuid },
    })

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
    const { roomId, text } = req.body as { roomId: number; text: string }

    if (!roomId) {
      res.send({ status: 'Fail', message: 'roomId required', data: null })
      return
    }

    const chatId = await abortChatProcess(userId, Number(roomId))
    if (chatId && typeof text === 'string') {
      try { await updateChatResponsePartial(chatId, text) } catch {}
    }

    res.send({ status: 'Success', message: 'OK', data: null })
  }
  catch (error: any) {
    res.send({ status: 'Fail', message: error?.message ?? 'Abort failed', data: null })
  }
})

router.post('/chat-process-status', auth, async (req, res) => {
  try {
    const userId = req.headers.userId as string
    const roomId = Number((req.body as any)?.roomId)

    if (!roomId) {
      res.send({ status: 'Success', message: '', data: { isProcessing: false, roomId: null, uuid: null, messageId: null } })
      return
    }

    const key = jobKey(userId, roomId)
    const job = jobStates.get(key)

    if (job?.status === 'running') {
      res.send({
        status: 'Success',
        message: '',
        data: {
          isProcessing: true,
          roomId: job.roomId,
          uuid: job.uuid,
          messageId: job.chatId,
        },
      })
      return
    }

    res.send({
      status: 'Success',
      message: '',
      data: { isProcessing: false, roomId, uuid: job?.uuid ?? null, messageId: job?.chatId ?? null },
    })
  }
  catch (error: any) {
    res.send({ status: 'Fail', message: error.message, data: null })
  }
})
