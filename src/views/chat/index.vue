<script setup lang='ts'>
import type { Ref } from 'vue'
import { computed, defineAsyncComponent, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { storeToRefs } from 'pinia'
import type { MessageReactive, UploadFileInfo } from 'naive-ui'
import { NAutoComplete, NButton, NInput, NSelect, NSlider, NSpace, NSpin, NUpload, useDialog, useMessage } from 'naive-ui'
import html2canvas from 'html2canvas'
import { Message } from './components'
import { useScroll } from './hooks/useScroll'
import { useChat } from './hooks/useChat'
import HeaderComponent from './components/Header/index.vue'
import { HoverButton, SvgIcon } from '@/components/common'
import { useBasicLayout } from '@/hooks/useBasicLayout'
import { useAuthStore, useChatStore, usePromptStore, useUserStore } from '@/store'
import { fetchChatAPIProcess, fetchChatResponseoHistory, fetchChatStopResponding } from '@/api'
import { t } from '@/locales'
import { debounce } from '@/utils/functions/debounce'
import IconPrompt from '@/icons/Prompt.vue'
import { get, post } from '@/utils/request'

const Prompt = defineAsyncComponent(() => import('@/components/common/Setting/Prompt.vue'))

let controller = new AbortController()

const route = useRoute()
const dialog = useDialog()
const ms = useMessage()
const authStore = useAuthStore()
const userStore = useUserStore()
const chatStore = useChatStore()

const { isMobile } = useBasicLayout()
const { addChat, updateChat, updateChatSome } = useChat()
const { scrollRef, scrollTo, scrollToBottom, scrollToBottomIfAtBottom } = useScroll()

const routeUuid = String((route.params as any)?.uuid ?? '')

const currentChatHistory = computed(() => chatStore.getChatHistoryByCurrentActive)
const usingContext = computed(() => currentChatHistory?.value?.usingContext ?? true)

const dataSources = computed(() => {
  const id = chatStore.active ?? (routeUuid ? Number(routeUuid) : undefined)
  if (typeof id !== 'number' || Number.isNaN(id)) return []
  return chatStore.getChatByUuid(id)
})

const conversationList = computed(() => dataSources.value.filter(item => (!item.inversion && !!item.conversationOptions)))

const prompt = ref<string>('')
const firstLoading = ref<boolean>(false)
const loading = ref<boolean>(false)
const inputRef = ref<Ref | null>(null)
const showPrompt = ref(false)
const nowSelectChatModel = ref<string | null>(null)
const currentChatModel = computed(() => nowSelectChatModel.value ?? currentChatHistory.value?.chatModel ?? userStore.userInfo.config.chatModel)

// ✅ 当前 room 正在生成的 uuid（统一用 undefined，不用 null）
const processingUuidRef = ref<number | undefined>(undefined)

const groupedChatModelOptions = computed(() => {
  const session = authStore.session
  const base = session?.chatModels ?? []
  if (!base.length) return []

  const geminiSet = new Set(session?.geminiChatModels ?? [])
  const nonStreamSet = new Set(session?.nonStreamChatModels ?? [])

  const gemini: any[] = []
  const nonStream: any[] = []
  const others: any[] = []

  for (const opt of base) {
    if (geminiSet.has(opt.value)) gemini.push(opt)
    else if (nonStreamSet.has(opt.value)) nonStream.push(opt)
    else others.push(opt)
  }

  const groups: any[] = []
  if (gemini.length) groups.push({ type: 'group', label: 'Gemini', key: 'g_gemini', children: gemini })
  if (nonStream.length) groups.push({ type: 'group', label: '非流式/图片模型', key: 'g_nonstream', children: nonStream })
  if (others.length) groups.push({ type: 'group', label: '其它模型', key: 'g_others', children: others })
  return groups
})

const currentNavIndexRef = ref<number>(-1)
const isVisionModel = ref(true)

/**
 * ✅ 联网搜索开关（前端控制）
 * - true：后端会做“模型判断是否需要搜索 + 多轮调整关键词搜索”
 * - false：不启用
 */
const searchMode = ref(false)
/**
 * ✅ 后台是否允许联网搜索（管理员配置）
 * - true：用户可以使用地球按钮
 * - false：地球按钮禁用
 */
const webSearchAllowed = computed(() => authStore.session?.webSearchEnabled === true)

let loadingms: MessageReactive
let allmsg: MessageReactive
let prevScrollTop: number

const promptStore = usePromptStore()
const { promptList: promptTemplate } = storeToRefs<any>(promptStore)

const imageUploadFileKeysRef = ref<string[]>([])
const textUploadFileKeysRef = ref<string[]>([])

// ✅ 浏览器端 interval 类型
let pollInterval: ReturnType<typeof setInterval> | null = null

function getCurrentRoomId(): number | undefined {
  const id = chatStore.active ?? (routeUuid ? Number(routeUuid) : undefined)
  if (typeof id !== 'number' || Number.isNaN(id)) return undefined
  return id
}

function inferLatestUuidFromUI(): number | undefined {
  if (!dataSources.value.length) return undefined
  return dataSources.value[dataSources.value.length - 1].uuid ?? undefined
}

async function fetchRoomProcessStatus(roomId: number) {
  return await post<{
    isProcessing: boolean
    roomId: number | null
    uuid: number | null
    messageId: string | null
  }>({
    url: '/chat-process-status',
    data: { roomId },
  })
}

/**
 * ✅ 确保“本轮 jobUuid 对应的 assistant 消息”存在
 * - 如果已有：返回其 index
 * - 如果没有且最后一条是 user：补一个占位 assistant（uuid=jobUuid）
 * - 否则：返回最后一条 index（不强行插入，避免重复空白）
 */
function ensureAssistantPlaceholder(roomId: number, jobUuid: number): number | undefined {
  const existedIndex = dataSources.value.findIndex(m => !m.inversion && m.uuid === jobUuid)
  if (existedIndex !== -1)
    return existedIndex

  const lastIndex = dataSources.value.length - 1
  if (lastIndex < 0)
    return undefined

  const lastItem = dataSources.value[lastIndex]
  if (lastItem?.inversion) {
    addChat(roomId, {
      uuid: jobUuid,
      dateTime: new Date().toLocaleString(),
      text: '',
      loading: true,
      inversion: false,
      error: false,
      conversationOptions: null,
      requestOptions: { prompt: '...', options: null },
    })
    return dataSources.value.length - 1
  }

  // 最后一条是 assistant，但 uuid 不匹配：不再插入，防止叠两个空白
  return lastIndex
}

async function fetchLatestAndUpdateUI(roomId: number, finalize = false) {
  const u: number | undefined = (processingUuidRef.value ?? inferLatestUuidFromUI()) ?? undefined
  if (u === undefined) return

  try {
    const r = await get<{
      text: string
      isProcessing: boolean
      conversationOptions: any
      usage?: any
      chatId?: string
    }>({
      url: '/chat-latest',
      data: { roomId, uuid: u },
    })

    const d: any = r.data

    // ✅ 优先更新“jobUuid 对应的 assistant 消息”，如果不存在才看最后一条
    let targetIndex = dataSources.value.findIndex(m => !m.inversion && m.uuid === u)
    if (targetIndex === -1) {
      // 如果最后一条是 user，补一个占位（避免更新不到）
      const idx = ensureAssistantPlaceholder(roomId, u)
      if (idx === undefined) return
      targetIndex = idx
    }

    // 只更新 assistant
    if (!dataSources.value[targetIndex]?.inversion) {
      updateChatSome(roomId, targetIndex, {
        text: d.text ?? '',
        loading: finalize ? false : true,
        conversationOptions: d.conversationOptions ?? dataSources.value[targetIndex].conversationOptions ?? null,
        usage: d.usage ?? dataSources.value[targetIndex].usage,
      })
    }

    if (finalize) {
      loading.value = false
      processingUuidRef.value = undefined
      stopPolling()
    }

    scrollToBottomIfAtBottom()
  }
  catch {
    // ignore
  }
}

function startPolling() {
  stopPolling()
  pollInterval = setInterval(async () => {
    try {
      const roomId = getCurrentRoomId()
      if (!roomId) return

      const { data } = await fetchRoomProcessStatus(roomId)

      if (data.isProcessing) {
        loading.value = true

        const jobUuid: number | undefined = (data.uuid ?? processingUuidRef.value ?? inferLatestUuidFromUI()) ?? undefined
        processingUuidRef.value = jobUuid

        // ✅ 历史还没加载出来：不要插入占位，直接等下一轮
        if (!dataSources.value.length) return

        // ✅ 如果已知 jobUuid，确保占位存在（必要时补一条）
        if (jobUuid !== undefined) {
          const idx = ensureAssistantPlaceholder(roomId, jobUuid)
          if (idx !== undefined && !dataSources.value[idx]?.inversion && !dataSources.value[idx].loading)
            updateChatSome(roomId, idx, { loading: true })
        }

        await fetchLatestAndUpdateUI(roomId, false)
      }
      else {
        // done
        loading.value = false
        stopPolling()
        await fetchLatestAndUpdateUI(roomId, true)
      }
    }
    catch (e) {
      console.error(e)
      stopPolling()
    }
  }, 2000)
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

/**
 * ✅ 修复：切回房间时不要额外插入“空白 assistant”导致重复
 */
async function checkProcessStatus() {
  try {
    stopPolling()
    loading.value = false
    processingUuidRef.value = undefined

    const roomId = getCurrentRoomId()
    if (!roomId) return

    const { data } = await fetchRoomProcessStatus(roomId)

    if (!data.isProcessing) {
      loading.value = false
      processingUuidRef.value = undefined
      return
    }

    loading.value = true

    // ✅ 把后端的 number|null 统一转成 number|undefined
    const jobUuid: number | undefined = (data.uuid ?? inferLatestUuidFromUI()) ?? undefined
    processingUuidRef.value = jobUuid

    // ✅ 历史尚未加载：不插入占位，直接开始轮询
    if (!dataSources.value.length) {
      startPolling()
      return
    }

    // ✅ 没有 jobUuid（极少数）：直接轮询等待后端落库
    if (jobUuid === undefined) {
      startPolling()
      return
    }

    // ✅ 已经存在该 jobUuid 的 assistant：只标记 loading
    const existingAssistantIndex = dataSources.value.findIndex(m => !m.inversion && m.uuid === jobUuid)
    if (existingAssistantIndex !== -1) {
      if (!dataSources.value[existingAssistantIndex].loading)
        updateChatSome(roomId, existingAssistantIndex, { loading: true })

      await fetchLatestAndUpdateUI(roomId, false)
      startPolling()
      return
    }

    // ✅ 只有最后一条是 user 时才补占位 assistant（uuid=jobUuid）
    const lastIndex = dataSources.value.length - 1
    const lastItem = dataSources.value[lastIndex]
    if (lastItem?.inversion) {
      addChat(roomId, {
        uuid: jobUuid,
        dateTime: new Date().toLocaleString(),
        text: '',
        loading: true,
        inversion: false,
        error: false,
        conversationOptions: null,
        requestOptions: { prompt: '...', options: null },
      })
      scrollToBottom()
    }
    else {
      if (!lastItem?.loading)
        updateChatSome(roomId, lastIndex, { loading: true })
    }

    await fetchLatestAndUpdateUI(roomId, false)
    startPolling()
  }
  catch (e) {
    console.error(e)
  }
}

const handleLoadingChain = async () => {
  const roomId = getCurrentRoomId()
  if (!roomId) return

  // 1) 切换房间时先停止旧轮询
  stopPolling()

  // 2) 拉取历史
  await chatStore.syncChat(
    { uuid: roomId } as Chat.History,
    undefined,
    () => {
      firstLoading.value = false
      nextTick(() => {
        const scrollRefDom = document.querySelector('#scrollRef') as HTMLElement | null
        if (scrollRefDom) scrollRefDom.scrollTop = scrollRefDom.scrollHeight
      })
      if (inputRef.value && !isMobile.value)
        (inputRef.value as any)?.focus()

      // 3) 历史拉完后，检查该房间是否有运行任务
      checkProcessStatus()
    },
  )
}

const debouncedLoad = debounce(handleLoadingChain, 200)

async function onConversation() {
  let message = prompt.value
  if (!message || message.trim() === '') return
  if (loading.value) return

  const roomId = getCurrentRoomId()
  if (!roomId) return

  if (nowSelectChatModel.value && currentChatHistory.value)
    currentChatHistory.value.chatModel = nowSelectChatModel.value

  const uploadFileKeys = isVisionModel.value
    ? [
        ...imageUploadFileKeysRef.value.map(k => `img:${k}`),
        ...textUploadFileKeysRef.value.map(k => `txt:${k}`),
      ]
    : []

  imageUploadFileKeysRef.value = []
  textUploadFileKeysRef.value = []

  controller = new AbortController()

  const chatUuid = Date.now()
  processingUuidRef.value = chatUuid

  addChat(roomId, {
    uuid: chatUuid,
    dateTime: new Date().toLocaleString(),
    text: message,
    images: uploadFileKeys,
    inversion: true,
    error: false,
    conversationOptions: null,
    requestOptions: { prompt: message, options: null },
  })
  scrollToBottom()

  loading.value = true
  prompt.value = ''

  let options: Chat.ConversationRequest = {}
  const lastContext = conversationList.value[conversationList.value.length - 1]?.conversationOptions
  if (lastContext && usingContext.value)
    options = { ...lastContext }

  addChat(roomId, {
    uuid: chatUuid,
    dateTime: new Date().toLocaleString(),
    text: '',
    loading: true,
    inversion: false,
    error: false,
    conversationOptions: null,
    requestOptions: { prompt: message, options: { ...options } },
  })
  scrollToBottom()

  try {
    await fetchChatAPIProcess<{ jobId: string }>({
      roomId,
      uuid: chatUuid,
      regenerate: false,
      prompt: message,
      searchMode: searchMode.value,
      uploadFileKeys,
      options,
      signal: controller.signal,
    })
    startPolling()
  }
  catch (error: any) {
    const errorMessage = error?.message ?? t('common.wrong')
    updateChat(roomId, dataSources.value.length - 1, {
      dateTime: new Date().toLocaleString(),
      text: errorMessage,
      inversion: false,
      error: true,
      loading: false,
      conversationOptions: null,
      requestOptions: { prompt: message, options: { ...options } },
    })
    loading.value = false
    stopPolling()
    processingUuidRef.value = undefined
  }
}

/**
 * ✅ Stop Responding：保证传 roomId，且立即停止轮询/结束 loading
 */
async function handleStop() {
  if (!loading.value) return

  const roomId = getCurrentRoomId()
  if (!roomId) return

  // 1) abort 前端请求（如果还没返回）
  controller.abort()

  // 2) 取当前 UI 已输出文本
  const u = processingUuidRef.value
  const assistantIndex = (u !== undefined)
    ? dataSources.value.findIndex(m => !m.inversion && m.uuid === u)
    : -1

  const lastIndex = dataSources.value.length - 1
  const currentAssistantIndex = assistantIndex !== -1
    ? assistantIndex
    : (lastIndex >= 0 && !dataSources.value[lastIndex].inversion ? lastIndex : -1)

  const currentText = currentAssistantIndex >= 0 ? (dataSources.value[currentAssistantIndex].text ?? '') : ''

  try {
    await fetchChatStopResponding(roomId, currentText)
  }
  catch (e) {
    console.error('[StopResponding] abort failed:', e)
  }
  finally {
    // 3) 立即结束前端状态
    stopPolling()
    loading.value = false
    processingUuidRef.value = undefined

    if (currentAssistantIndex >= 0 && !dataSources.value[currentAssistantIndex].inversion)
      updateChatSome(roomId, currentAssistantIndex, { loading: false })

    // 4) 稍后再拉一次最终内容（后端可能已落库 partial）
    setTimeout(() => {
      void fetchLatestAndUpdateUI(roomId, true)
    }, 300)
  }
}

async function onRegenerate(index: number) {
  if (loading.value) return
  const roomId = getCurrentRoomId()
  if (!roomId) return

  controller = new AbortController()

  const { requestOptions } = dataSources.value[index]
  let responseCount = dataSources.value[index].responseCount || 1
  responseCount++

  const message = requestOptions?.prompt ?? ''
  let options: Chat.ConversationRequest = {}
  if (requestOptions.options) options = { ...requestOptions.options }

  let uploadFileKeys: string[] = []
  if (index > 0) {
    const previousMessage = dataSources.value[index - 1]
    if (previousMessage && previousMessage.inversion && previousMessage.images && previousMessage.images.length > 0)
      uploadFileKeys = previousMessage.images
  }

  loading.value = true
  const chatUuid = dataSources.value[index].uuid || Date.now()
  processingUuidRef.value = chatUuid

  updateChat(roomId, index, {
    dateTime: new Date().toLocaleString(),
    text: '',
    inversion: false,
    responseCount,
    error: false,
    loading: true,
    conversationOptions: null,
    requestOptions: { prompt: message, options: { ...options } },
  })

  try {
    await fetchChatAPIProcess<{ jobId: string }>({
      roomId,
      uuid: chatUuid,
      regenerate: true,
      prompt: message,
      searchMode: searchMode.value,
      uploadFileKeys,
      options,
      signal: controller.signal,
    })
    startPolling()
  }
  catch (error: any) {
    const errorMessage = error?.message ?? t('common.wrong')
    updateChat(roomId, index, {
      dateTime: new Date().toLocaleString(),
      text: errorMessage,
      inversion: false,
      responseCount,
      error: true,
      loading: false,
      conversationOptions: null,
      requestOptions: { prompt: message, options: { ...options } },
    })
    loading.value = false
    stopPolling()
    processingUuidRef.value = undefined
  }
}

async function onResponseHistory(index: number, historyIndex: number) {
  const roomId = getCurrentRoomId()
  if (!roomId) return
  const chat = (await fetchChatResponseoHistory(roomId, dataSources.value[index].uuid || Date.now(), historyIndex)).data
  updateChat(roomId, index, {
    dateTime: chat.dateTime,
    text: chat.text,
    inversion: false,
    responseCount: chat.responseCount,
    error: true,
    loading: false,
    conversationOptions: chat.conversationOptions,
    requestOptions: { prompt: chat.requestOptions.prompt, options: { ...chat.requestOptions.options } },
    usage: chat.usage,
  })
}

function handleExport() {
  if (loading.value) return
  const d = dialog.warning({
    title: t('chat.exportImage'),
    content: t('chat.exportImageConfirm'),
    positiveText: t('common.yes'),
    negativeText: t('common.no'),
    onPositiveClick: async () => {
      try {
        d.loading = true
        const ele = document.getElementById('image-wrapper')
        const canvas = await html2canvas(ele as HTMLDivElement, { useCORS: true })
        const imgUrl = canvas.toDataURL('image/png')
        const tempLink = document.createElement('a')
        tempLink.style.display = 'none'
        tempLink.href = imgUrl
        tempLink.setAttribute('download', 'chat-shot.png')
        if (typeof tempLink.download === 'undefined') tempLink.setAttribute('target', '_blank')
        document.body.appendChild(tempLink)
        tempLink.click()
        document.body.removeChild(tempLink)
        window.URL.revokeObjectURL(imgUrl)
        ms.success(t('chat.exportSuccess'))
      }
      catch {
        ms.error(t('chat.exportFailed'))
      }
      finally {
        d.loading = false
      }
    },
  })
}

function handleDelete(index: number, fast: boolean) {
  if (loading.value) return
  const roomId = getCurrentRoomId()
  if (!roomId) return

  if (fast === true) {
    chatStore.deleteChatByUuid(roomId, index)
  }
  else {
    dialog.warning({
      title: t('chat.deleteMessage'),
      content: t('chat.deleteMessageConfirm'),
      positiveText: t('common.yes'),
      negativeText: t('common.no'),
      onPositiveClick: () => {
        chatStore.deleteChatByUuid(roomId, index)
      },
    })
  }
}

function updateCurrentNavIndex(_index: number, newIndex: number) {
  currentNavIndexRef.value = newIndex
}

function handleClear() {
  if (loading.value) return
  const roomId = getCurrentRoomId()
  if (!roomId) return

  dialog.warning({
    title: t('chat.clearChat'),
    content: t('chat.clearChatConfirm'),
    positiveText: t('common.yes'),
    negativeText: t('common.no'),
    onPositiveClick: () => {
      chatStore.clearChatByUuid(roomId)
    },
  })
}

function handleEnter(event: KeyboardEvent) {
  if (!isMobile.value) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onConversation()
    }
  }
  else {
    if (event.key === 'Enter' && event.ctrlKey) {
      event.preventDefault()
      onConversation()
    }
  }
}

async function loadMoreMessage(event: any) {
  const roomId = getCurrentRoomId()
  if (!roomId) return

  const chatIndex = chatStore.chat.findIndex(d => d.uuid === roomId)
  if (chatIndex <= -1 || chatStore.chat[chatIndex].data.length <= 0) return

  const scrollPosition = event.target.scrollHeight - event.target.scrollTop
  const lastId = chatStore.chat[chatIndex].data[0].uuid

  await chatStore.syncChat(
    { uuid: roomId } as Chat.History,
    lastId,
    () => {
      loadingms && loadingms.destroy()
      nextTick(() => scrollTo(event.target.scrollHeight - scrollPosition))
    },
    () => {
      loadingms = ms.loading('加载中...', { duration: 0 })
    },
    () => {
      allmsg && allmsg.destroy()
      allmsg = ms.warning('没有更多了', { duration: 1000 })
    },
  )
}

const handleLoadMoreMessage = debounce(loadMoreMessage, 300)

async function handleScroll(event: any) {
  const scrollTop = event.target.scrollTop
  if (scrollTop < 50 && (scrollTop < prevScrollTop || prevScrollTop === undefined))
    handleLoadMoreMessage(event)
  prevScrollTop = scrollTop
}

async function handleToggleUsingContext() {
  const roomId = getCurrentRoomId()
  if (!currentChatHistory.value || !roomId) return
  currentChatHistory.value.usingContext = !currentChatHistory.value.usingContext
  chatStore.setUsingContext(currentChatHistory.value.usingContext, roomId)
  if (currentChatHistory.value.usingContext) ms.success(t('chat.turnOnContext'))
  else ms.warning(t('chat.turnOffContext'))
}

const searchOptions = computed(() => {
  if (prompt.value.startsWith('/')) {
    return promptTemplate.value
      .filter((item: { title: string }) => item.title.toLowerCase().includes(prompt.value.substring(1).toLowerCase()))
      .map((obj: { value: any }) => ({ label: obj.value, value: obj.value }))
  }
  return []
})

function renderOption(option: { label: string }) {
  for (const i of promptTemplate.value) {
    if (i.value === option.label) return [i.title]
  }
  return []
}

const placeholder = computed(() => (isMobile.value ? t('chat.placeholderMobile') : t('chat.placeholder')))
const buttonDisabled = computed(() => loading.value || !prompt.value || prompt.value.trim() === '')
const footerClass = computed(() => (isMobile.value ? ['sticky', 'left-0', 'bottom-0', 'right-0', 'p-2', 'pr-3', 'overflow-hidden'] : ['p-4']))

async function handleSyncChatModel(chatModel: string) {
  const roomId = getCurrentRoomId()
  if (!roomId) return
  nowSelectChatModel.value = chatModel
  await chatStore.setChatModel(chatModel, roomId)
}

function formatTooltip(value: number) {
  return `${t('setting.maxContextCount')}: ${value}`
}

function handleBeforeUpload(data: { file: UploadFileInfo }) {
  if (data.file.file?.size && data.file.file.size / 1024 / 1024 > 100) {
    ms.error('文件大小不能超过 100MB')
    return false
  }
  return true
}

function handleFinishImage(options: { file: UploadFileInfo; event?: ProgressEvent }) {
  if (options.file.status === 'finished') {
    const response = (options.event?.target as XMLHttpRequest).response
    const key = `${response.data.fileKey}`
    imageUploadFileKeysRef.value.push(key)
    ms.success('图片上传成功')
  }
}

function handleFinishText(options: { file: UploadFileInfo; event?: ProgressEvent }) {
  if (options.file.status === 'finished') {
    const response = (options.event?.target as XMLHttpRequest).response
    const key = `${response.data.fileKey}`
    textUploadFileKeysRef.value.push(key)
    ms.success('文件上传成功')
  }
}

function handleDeleteImageFile(index: number) {
  imageUploadFileKeysRef.value.splice(index, 1)
}

function handleDeleteTextFile(index: number) {
  textUploadFileKeysRef.value.splice(index, 1)
}

const uploadHeaders = computed(() => {
  const token = useAuthStore().token
  return { Authorization: `Bearer ${token}` }
})

onMounted(async () => {
  firstLoading.value = true
  debouncedLoad()
})

watch(
  () => chatStore.active,
  () => {
    // 切换房间：停止旧轮询，重置状态
    stopPolling()
    processingUuidRef.value = undefined
    loading.value = false

    if (chatStore.active) {
      firstLoading.value = true
      debouncedLoad()
    }
  },
)

onUnmounted(() => stopPolling())
</script>
<template>
  <div class="flex flex-col w-full h-full">
    <HeaderComponent
      v-if="isMobile"
      :using-context="usingContext"
      :show-prompt="showPrompt"
      @export="handleExport"
      @toggle-using-context="handleToggleUsingContext"
      @toggle-show-prompt="showPrompt = true"
    />
    <main class="flex-1 overflow-hidden">
      <div id="scrollRef" ref="scrollRef" class="h-full overflow-hidden overflow-y-auto" @scroll="handleScroll">
        <div id="image-wrapper" class="w-full max-w-screen-xl m-auto dark:bg-[#101014]" :class="[isMobile ? 'p-2' : 'p-4']">
          <NSpin :show="firstLoading">
            <template v-if="!dataSources.length">
              <div class="flex items-center justify-center mt-4 text-center text-neutral-300">
                <SvgIcon icon="ri:bubble-chart-fill" class="mr-2 text-3xl" />
                <span>Aha~</span>
              </div>
            </template>
            <template v-else>
              <div>
                <Message
                  v-for="(item, index) of dataSources"
                  :key="index"
                  :index="index"
                  :current-nav-index="currentNavIndexRef"
                  :date-time="item.dateTime"
                  :text="item.text"
                  :images="item.images"
                  :inversion="item.inversion"
                  :response-count="item.responseCount"
                  :usage="item && item.usage || undefined"
                  :error="item.error"
                  :loading="item.loading"
                  @regenerate="onRegenerate(index)"
                  @update-current-nav-index="(itemId: number) => updateCurrentNavIndex(index, itemId)"
                  @delete="(fast) => handleDelete(index, fast)"
                  @response-history="(ev) => onResponseHistory(index, ev)"
                />
                <div class="sticky bottom-0 left-0 flex justify-center">
                  <NButton v-if="loading" type="warning" @click="handleStop">
                    <template #icon><SvgIcon icon="ri:stop-circle-line" /></template>
                    Stop Responding
                  </NButton>
                </div>
              </div>
            </template>
          </NSpin>
        </div>
      </div>
    </main>

    <footer :class="footerClass">
      <div class="w-full max-w-screen-xl m-auto">
        <NSpace vertical>
          <div v-if="imageUploadFileKeysRef.length > 0" class="flex flex-wrap items-center gap-2 mb-2">
            <div v-for="(key, index) in imageUploadFileKeysRef" :key="`img-${index}`" class="flex items-center px-2 py-1 text-xs bg-gray-100 rounded dark:bg-neutral-700">
              <SvgIcon icon="ri:image-line" class="mr-1" />
              <span class="truncate max-w-[150px]">{{ key }}</span>
              <button class="ml-1 text-red-500 hover:text-red-700" @click="handleDeleteImageFile(index)">
                <SvgIcon icon="ri:close-line" />
              </button>
            </div>
          </div>

          <div v-if="textUploadFileKeysRef.length > 0" class="flex flex-wrap items-center gap-2 mb-2">
            <div v-for="(key, index) in textUploadFileKeysRef" :key="`txt-${index}`" class="flex items-center px-2 py-1 text-xs bg-gray-100 rounded dark:bg-neutral-700">
              <SvgIcon icon="ri:file-text-line" class="mr-1" />
              <span class="truncate max-w-[150px]">{{ key }}</span>
              <button class="ml-1 text-red-500 hover:text-red-700" @click="handleDeleteTextFile(index)">
                <SvgIcon icon="ri:close-line" />
              </button>
            </div>
          </div>

          <div class="flex items-center space-x-2">
            <div>
              <NUpload
                action="/api/upload-image"
                multiple
                :max="100"
                :headers="uploadHeaders"
                :show-file-list="false"
                response-type="json"
                accept="image/*"
                @finish="handleFinishImage"
                @before-upload="handleBeforeUpload"
              >
                <div class="flex items-center justify-center h-10 px-2 transition rounded-md hover:bg-neutral-100 dark:hover:bg-[#414755] cursor-pointer">
                  <span class="text-xl text-[#4f555e] dark:text-white"><SvgIcon icon="ri:image-add-line" /></span>
                </div>
              </NUpload>
            </div>

            <div>
              <NUpload
                action="/api/upload-image"
                multiple
                :max="100"
                :headers="uploadHeaders"
                :show-file-list="false"
                response-type="json"
                accept=".txt,.md,.json,.csv,.js,.ts,.py,.java,.html,.css,.xml,.yml,.yaml,.log,.ini,.config"
                @finish="handleFinishText"
                @before-upload="handleBeforeUpload"
              >
                <div class="flex items-center justify-center h-10 px-2 transition rounded-md hover:bg-neutral-100 dark:hover:bg-[#414755] cursor-pointer">
                  <span class="text-xl text-[#4f555e] dark:text-white"><SvgIcon icon="ri:attachment-2" /></span>
                </div>
              </NUpload>
            </div>

            <!-- ✅ 联网搜索开关（高亮逻辑同"上下文"按钮；后台未启用则禁用） -->
            <HoverButton
              :tooltip="!webSearchAllowed
                ? '联网搜索：管理员未启用'
                : searchMode
                  ? '联网搜索：开（模型将自动决定是否搜索并可多轮调整关键词）'
                  : '联网搜索：关'"
              :class="{
                'text-[#4b9e5f]': searchMode && webSearchAllowed,
                'text-[#a8071a]': !searchMode && webSearchAllowed,
                'opacity-40 cursor-not-allowed': !webSearchAllowed,
              }"
              @click="webSearchAllowed && (searchMode = !searchMode)"
            >
              <span class="text-xl">
                <SvgIcon icon="ri:globe-line" />
              </span>
            </HoverButton>

            <HoverButton @click="handleClear">
              <span class="text-xl text-[#4f555e] dark:text-white"><SvgIcon icon="ri:delete-bin-line" /></span>
            </HoverButton>

            <HoverButton v-if="!isMobile" @click="handleExport">
              <span class="text-xl text-[#4f555e] dark:text-white"><SvgIcon icon="ri:download-2-line" /></span>
            </HoverButton>

            <HoverButton v-if="!isMobile" @click="showPrompt = true">
              <span class="text-xl text-[#4f555e] dark:text-white"><IconPrompt class="w-[20px] m-auto" /></span>
            </HoverButton>

            <HoverButton
              v-if="!isMobile"
              :tooltip="usingContext ? $t('chat.clickTurnOffContext') : $t('chat.clickTurnOnContext')"
              :class="{ 'text-[#4b9e5f]': usingContext, 'text-[#a8071a]': !usingContext }"
              @click="handleToggleUsingContext"
            >
              <span class="text-xl"><SvgIcon icon="ri:chat-history-line" /></span>
            </HoverButton>

            <NSelect
              style="width: 250px"
              :value="currentChatModel"
              :options="groupedChatModelOptions"
              :disabled="!!authStore.session?.auth && !authStore.token && !authStore.session?.authProxyEnabled"
              @update-value="(val) => handleSyncChatModel(val)"
            />

            <NSlider
              v-model:value="userStore.userInfo.advanced.maxContextCount"
              :max="100"
              :min="0"
              :step="1"
              style="width: 88px"
              :format-tooltip="formatTooltip"
              @update:value="() => { userStore.updateSetting(false) }"
            />
          </div>

          <div class="flex items-center justify-between space-x-2">
            <NAutoComplete v-model:value="prompt" :options="searchOptions" :render-label="renderOption">
              <template #default="{ handleInput, handleBlur, handleFocus }">
                <NInput
                  ref="inputRef"
                  v-model:value="prompt"
                  :disabled="!!authStore.session?.auth && !authStore.token && !authStore.session?.authProxyEnabled"
                  type="textarea"
                  :placeholder="placeholder"
                  :autosize="{ minRows: isMobile ? 1 : 4, maxRows: isMobile ? 4 : 8 }"
                  @input="handleInput"
                  @focus="handleFocus"
                  @blur="handleBlur"
                  @keypress="handleEnter"
                />
              </template>
            </NAutoComplete>

            <NButton type="primary" :disabled="buttonDisabled" @click="onConversation">
              <template #icon><span class="dark:text-black"><SvgIcon icon="ri:send-plane-fill" /></span></template>
            </NButton>
          </div>
        </NSpace>
      </div>
    </footer>

    <Prompt v-if="showPrompt" v-model:roomId="routeUuid" v-model:visible="showPrompt" />
  </div>
</template>
