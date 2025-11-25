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
import {
  fetchChatAPIProcess,
  fetchChatProcessStatus,
  fetchChatResponseoHistory,
  fetchChatStopResponding,
} from '@/api'
import { t } from '@/locales'
import { debounce } from '@/utils/functions/debounce'
import IconPrompt from '@/icons/Prompt.vue'

const Prompt = defineAsyncComponent(() => import('@/components/common/Setting/Prompt.vue'))

let controller = new AbortController()
let lastChatInfo: any = { text: 'Stopped', id: null, conversationId: null }

const openLongReply = import.meta.env.VITE_GLOB_OPEN_LONG_REPLY === 'true'

const route = useRoute()
const dialog = useDialog()
const ms = useMessage()
const authStore = useAuthStore()
const userStore = useUserStore()
const chatStore = useChatStore()

const { isMobile } = useBasicLayout()
const { addChat, updateChat, updateChatSome, getChatByUuidAndIndex } = useChat()
const { scrollRef, scrollTo, scrollToBottom, scrollToBottomIfAtBottom } = useScroll()

// 从路由获取 UUID
const { uuid } = route.params as { uuid: string }

const currentChatHistory = computed(() => chatStore.getChatHistoryByCurrentActive)
const usingContext = computed(() => currentChatHistory?.value?.usingContext ?? true)
// 优先使用 store active，兜底使用路由 uuid
const dataSources = computed(() => {
  const id = chatStore.active ?? Number(uuid)
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

const currentNavIndexRef = ref<number>(-1)

const isVisionModel = ref(true)

let loadingms: MessageReactive
let allmsg: MessageReactive
let prevScrollTop: number

const promptStore = usePromptStore()
const { promptList: promptTemplate } = storeToRefs<any>(promptStore)

function handleSubmit() {
  onConversation()
}

const imageUploadFileKeysRef = ref<string[]>([])
const textUploadFileKeysRef = ref<string[]>([])

let pollInterval: NodeJS.Timeout | null = null

// ================== 核心修复逻辑 ==================

// 获取当前准确的 RoomId
function getCurrentRoomId(): number {
  // 优先取 store 里的活跃 ID，如果没有（比如刚刷新未 hydrates），取路由上的 uuid 转数字
  return chatStore.active ? chatStore.active : Number(uuid)
}

// 1. 状态检查 (Debug 版)
async function checkProcessStatus() {
  try {
    stopPolling() 
    const roomId = getCurrentRoomId()
    if (!roomId) {
      console.error('[Debug] CheckStatus: No RoomId found', roomId)
      return
    }

    console.log('[Debug] Checking status for Room:', roomId)

    const { data } = await fetchChatProcessStatus<{
      isProcessing: boolean
      roomId: number | null
      messageId: string | null
    }>()

    console.log('[Debug] Backend Status:', data)

    // 强转 string/number 对比，防止类型错误
    if (data.isProcessing && Number(data.roomId) === roomId) {
      console.log('[Debug] Resume loading state')
      loading.value = true
      
      lastChatInfo = {
        id: data.messageId,
        conversationId: null,
        text: '...',
      }

      const lastIndex = dataSources.value.length - 1
      const lastItem = dataSources.value[lastIndex]

      // 如果最后一条是用户发的消息，或者列表为空，说明回复还没存入数据库
      // 我们补一个假的“...”
      if (lastIndex < 0 || (lastItem && lastItem.inversion)) {
        console.log('[Debug] Appending fake loading bubble')
        addChat(
          roomId,
          {
            uuid: Date.now(),
            dateTime: new Date().toLocaleString(),
            text: '...',
            loading: true,
            inversion: false,
            error: false,
            conversationOptions: null,
            requestOptions: { prompt: '...', options: null },
          },
        )
        scrollToBottom()
      } else if (lastItem) {
        // 已经有回复了，强制转圈
        console.log('[Debug] Updating existing bubble loading=true')
        updateChatSome(roomId, lastIndex, { loading: true })
      }

      startPolling()
    } else {
      console.log('[Debug] No active process for this room')
    }
  } catch (error) {
    console.error('[Debug] Check process status failed:', error)
  }
}

// 2. 轮询逻辑 (Debug 版)
function startPolling() {
  stopPolling() // 防抖
  console.log('[Debug] Start Polling...')
  
  pollInterval = setInterval(async () => {
    try {
      const roomId = getCurrentRoomId()
      if (!roomId) return

      const { data } = await fetchChatProcessStatus<{
        isProcessing: boolean
        roomId: number | null
        messageId: string | null
      }>()

      // 切房间了，停
      if (Number(data.roomId) !== roomId) {
        // console.log('[Debug] Room changed, stop polling current.')
        // 此处不一定要 stop，因为可能后端还在生成上一个房间的，但前端无需展示
        // 只有当 data.roomId === null 或任务结束时才真正停
        if (!data.isProcessing) stopPolling()
        return
      }

      if (data.isProcessing) {
        loading.value = true
        // 维持 loading 状态
        const lastIndex = dataSources.value.length - 1
        if (lastIndex > -1 && !dataSources.value[lastIndex].inversion && !dataSources.value[lastIndex].loading) {
          updateChatSome(roomId, lastIndex, { loading: true })
        }
      } else {
        console.log('[Debug] Task finished, reloading history.')
        stopPolling()
        loading.value = false

        // 延迟重拉
        setTimeout(async () => {
          const chatIndex = chatStore.chat.findIndex(d => d.uuid === roomId)
          if (chatIndex > -1) chatStore.chat[chatIndex].data = [] 

          await chatStore.syncChat(
            { uuid: roomId } as Chat.History,
            undefined,
            () => { scrollToBottom() }
          )
        }, 1000)
      }
    } catch (error) {
      console.error('Poll error', error)
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

// 3. 串行加载链
const handleLoadingChain = async () => {
  const roomId = getCurrentRoomId()
  if (!roomId) return

  stopPolling()

  console.log('[Debug] Syncing chat history for:', roomId)
  await chatStore.syncChat(
    { uuid: roomId } as Chat.History,
    undefined,
    () => {
      firstLoading.value = false
      
      nextTick(() => {
        const scrollRefDom = document.querySelector('#scrollRef')
        if (scrollRefDom) scrollRefDom.scrollTop = scrollRefDom.scrollHeight
      })
      if (inputRef.value && !isMobile.value)
        (inputRef.value as any)?.focus()

      // [重要] 历史拉完后，去检查是否还在生成
      checkProcessStatus()
    }
  )
}

const debouncedLoad = debounce(handleLoadingChain, 200)

// ================== 发送消息 ==================
async function onConversation() {
  let message = prompt.value

  if (loading.value) return
  if (!message || message.trim() === '') return

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
  // 用户消息
  addChat(
    roomId,
    {
      uuid: chatUuid,
      dateTime: new Date().toLocaleString(),
      text: message,
      images: uploadFileKeys,
      inversion: true,
      error: false,
      conversationOptions: null,
      requestOptions: { prompt: message, options: null },
    },
  )
  scrollToBottom()

  loading.value = true
  prompt.value = ''

  let options: Chat.ConversationRequest = {}
  const lastContext = conversationList.value[conversationList.value.length - 1]?.conversationOptions

  if (lastContext && usingContext.value)
    options = { ...lastContext }

  // AI 占位消息
  addChat(
    roomId,
    {
      uuid: chatUuid,
      dateTime: new Date().toLocaleString(),
      text: '',
      loading: true, // 初始 loading
      inversion: false,
      error: false,
      conversationOptions: null,
      requestOptions: { prompt: message, options: { ...options } },
    },
  )
  scrollToBottom()

  try {
    let lastText = ''
    const fetchChatAPIOnce = async () => {
      await fetchChatAPIProcess<Chat.ConversationResponse>({
        roomId: roomId,
        uuid: chatUuid,
        prompt: message,
        uploadFileKeys,
        options,
        signal: controller.signal,
        onDownloadProgress: ({ event }) => {
          const xhr = event.target as XMLHttpRequest
          const { responseText } = xhr
          const lastIndex = responseText.lastIndexOf('\n', responseText.length - 2)
          let chunk = responseText
          if (lastIndex !== -1) chunk = responseText.substring(lastIndex)
          try {
            const data = JSON.parse(chunk)
            lastChatInfo = data
            const usage = (data.detail && data.detail.usage)
              ? {
                  completion_tokens: data.detail.usage.completion_tokens || null,
                  prompt_tokens: data.detail.usage.prompt_tokens || null,
                  total_tokens: data.detail.usage.total_tokens || null,
                  estimated: data.detail.usage.estimated || null,
                }
              : undefined
            updateChat(
              roomId,
              dataSources.value.length - 1,
              {
                dateTime: new Date().toLocaleString(),
                text: lastText + (data.text ?? ''),
                inversion: false,
                error: false,
                loading: true,
                conversationOptions: { conversationId: data.conversationId, parentMessageId: data.id },
                requestOptions: { prompt: message, options: { ...options } },
                usage,
              },
            )

            if (openLongReply && data.detail?.choices?.[0]?.finish_reason === 'length') {
              options.parentMessageId = data.id
              lastText = data.text
              message = ''
              return fetchChatAPIOnce()
            }
            scrollToBottomIfAtBottom()
          }
          catch (error) {}
        },
      })
      updateChatSome(roomId, dataSources.value.length - 1, { loading: false })
    }

    await fetchChatAPIOnce()
  }
  catch (error: any) {
    if (error.message === 'canceled') {
      // 主动取消（Stop），前端停 loading，后端通过 /chat-abort 停
      updateChatSome(roomId, dataSources.value.length - 1, { loading: false })
      scrollToBottomIfAtBottom()
      return
    }

    const errorMessage = error?.message ?? t('common.wrong')
    const currentChat = getChatByUuidAndIndex(roomId, dataSources.value.length - 1)
    const iso1model = currentChatModel.value?.includes('o1')
    
    if (currentChat?.text && currentChat.text !== '' && !iso1model) {
      updateChatSome(roomId, dataSources.value.length - 1, {
          text: `${currentChat.text}\n[${errorMessage}]`,
          error: false,
          loading: false,
        })
      return
    }
    if (currentChat?.text && currentChat.text !== '' && iso1model) {
      updateChatSome(roomId, dataSources.value.length - 1, {
          text: `${currentChat.text}`,
          error: false,
          loading: false,
        })
      return
    }

    updateChat(
      roomId,
      dataSources.value.length - 1,
      {
        dateTime: new Date().toLocaleString(),
        text: errorMessage,
        inversion: false,
        error: true,
        loading: false,
        conversationOptions: null,
        requestOptions: { prompt: message, options: { ...options } },
      },
    )
    scrollToBottomIfAtBottom()
  }
  finally {
    loading.value = false
  }
}

async function handleStop() {
  if (loading.value) {
    controller.abort()
    loading.value = false
    stopPolling()
    
    // 防御性校验
    if (lastChatInfo.id) {
      await fetchChatStopResponding(
        lastChatInfo.text || 'Stopped',
        lastChatInfo.id,
        lastChatInfo.conversationId,
      )
    }
  }
}

async function onRegenerate(index: number) {
  if (loading.value) return
  const roomId = getCurrentRoomId() // Use dynamic getter
  if (!roomId) return

  controller = new AbortController()

  const { requestOptions } = dataSources.value[index]
  let responseCount = dataSources.value[index].responseCount || 1
  responseCount++

  let message = requestOptions?.prompt ?? ''
  let options: Chat.ConversationRequest = {}
  if (requestOptions.options) options = { ...requestOptions.options }

  let uploadFileKeys: string[] = []
  if (index > 0) {
    const previousMessage = dataSources.value[index - 1]
    if (previousMessage && previousMessage.inversion && previousMessage.images && previousMessage.images.length > 0)
      uploadFileKeys = previousMessage.images
  }

  loading.value = true
  const chatUuid = dataSources.value[index].uuid
  
  updateChat(
    roomId,
    index,
    {
      dateTime: new Date().toLocaleString(),
      text: '', // Clear text for redraw
      inversion: false,
      responseCount,
      error: false,
      loading: true,
      conversationOptions: null,
      requestOptions: { prompt: message, options: { ...options } },
    },
  )

  try {
    let lastText = ''
    const fetchChatAPIOnce = async () => {
      await fetchChatAPIProcess<Chat.ConversationResponse>({
        roomId,
        uuid: chatUuid || Date.now(),
        regenerate: true,
        prompt: message,
        uploadFileKeys,
        options,
        signal: controller.signal,
        onDownloadProgress: ({ event }) => {
          const xhr = event.target as XMLHttpRequest
          const chunk = xhr.responseText
          const lastIndex = chunk.lastIndexOf('\n', chunk.length - 2)
          let realChunk = chunk
          if (lastIndex !== -1) realChunk = chunk.substring(lastIndex)
          try {
            const data = JSON.parse(realChunk)
            lastChatInfo = data
            const usage = (data.detail && data.detail.usage) ? data.detail.usage : undefined
            updateChat(
              roomId,
              index,
              {
                dateTime: new Date().toLocaleString(),
                text: lastText + (data.text ?? ''),
                inversion: false,
                responseCount,
                error: false,
                loading: true,
                conversationOptions: { conversationId: data.conversationId, parentMessageId: data.id },
                requestOptions: { prompt: message, options: { ...options } },
                usage,
              },
            )
            if (openLongReply && data.detail?.choices?.[0]?.finish_reason === 'length') {
              options.parentMessageId = data.id
              lastText = data.text
              message = ''
              return fetchChatAPIOnce()
            }
          } catch (error) {}
        },
      })
      updateChatSome(roomId, index, { loading: false })
    }
    await fetchChatAPIOnce()
  } catch (error: any) {
    if (error.message === 'canceled') {
      updateChatSome(roomId, index, { loading: false })
      return
    }
    const errorMessage = error?.message ?? t('common.wrong')
    updateChat(
      roomId,
      index,
      {
        dateTime: new Date().toLocaleString(),
        text: errorMessage,
        inversion: false,
        responseCount,
        error: true,
        loading: false,
        conversationOptions: null,
        requestOptions: { prompt: message, options: { ...options } },
      },
    )
  } finally {
    loading.value = false
  }
}

async function onResponseHistory(index: number, historyIndex: number) {
  const roomId = getCurrentRoomId()
  if (!roomId) return
  const chat = (await fetchChatResponseoHistory(roomId, dataSources.value[index].uuid || Date.now(), historyIndex)).data
  updateChat(
    roomId,
    index,
    {
      dateTime: chat.dateTime,
      text: chat.text,
      inversion: false,
      responseCount: chat.responseCount,
      error: true,
      loading: false,
      conversationOptions: chat.conversationOptions,
      requestOptions: { prompt: chat.requestOptions.prompt, options: { ...chat.requestOptions.options } },
      usage: chat.usage,
    },
  )
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
        d.loading = false
        ms.success(t('chat.exportSuccess'))
        Promise.resolve()
      } catch (error: any) {
        ms.error(t('chat.exportFailed'))
      } finally {
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
  } else {
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

function updateCurrentNavIndex(index: number, newIndex: number) {
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
      handleSubmit()
    }
  } else {
    if (event.key === 'Enter' && event.ctrlKey) {
      event.preventDefault()
      handleSubmit()
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

  await chatStore.syncChat({ uuid: roomId } as Chat.History, lastId, () => {
    loadingms && loadingms.destroy()
    nextTick(() => scrollTo(event.target.scrollHeight - scrollPosition))
  }, () => {
    loadingms = ms.loading('加载中...', { duration: 0 })
  }, () => {
    allmsg && allmsg.destroy()
    allmsg = ms.warning('没有更多了', { duration: 1000 })
  })
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
    return promptTemplate.value.filter((item: { title: string }) => item.title.toLowerCase().includes(prompt.value.substring(1).toLowerCase())).map((obj: { value: any }) => {
      return { label: obj.value, value: obj.value }
    })
  }
  return []
})

function renderOption(option: { label: string }) {
  for (const i of promptTemplate.value) {
    if (i.value === option.label) return [i.title]
  }
  return []
}

const placeholder = computed(() => {
  if (isMobile.value) return t('chat.placeholderMobile')
  return t('chat.placeholder')
})

const buttonDisabled = computed(() => {
  return loading.value || !prompt.value || prompt.value.trim() === ''
})

const footerClass = computed(() => {
  let classes = ['p-4']
  if (isMobile.value) classes = ['sticky', 'left-0', 'bottom-0', 'right-0', 'p-2', 'pr-3', 'overflow-hidden']
  return classes
})

async function handleSyncChatModel(chatModel: string) {
  const roomId = getCurrentRoomId()
  if (!roomId) return
  nowSelectChatModel.value = chatModel
  await chatStore.setChatModel(chatModel, roomId)
}

function formatTooltip(value: number) {
  return `${t('setting.maxContextCount')}: ${value}`
}

function handleBeforeUpload(data: { file: UploadFileInfo; fileList: UploadFileInfo[] }) {
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

// 生命周期：页面加载
onMounted(async () => {
  firstLoading.value = true  
  debouncedLoad() // 这里会先同步历史，然后调 checkProcessStatus

  if (authStore.token) {
    const chatModels = authStore.session?.chatModels
    if (chatModels != null && chatModels.filter(d => d.value === userStore.userInfo.config.chatModel).length <= 0)
      ms.error('你选择的模型已不存在，请重新选择 | The selected model not exists, please choose again.', { duration: 7000 })
  }
})

// 监听房间切换
watch(
  () => chatStore.active,
  () => {
    stopPolling()
    if (chatStore.active) {
      firstLoading.value = true
      debouncedLoad()
    }
  },
)

onUnmounted(() => {
  stopPolling()
})
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
      <div
        id="scrollRef"
        ref="scrollRef"
        class="h-full overflow-hidden overflow-y-auto"
        @scroll="handleScroll"
      >
        <div
          id="image-wrapper"
          class="w-full max-w-screen-xl m-auto dark:bg-[#101014]"
          :class="[isMobile ? 'p-2' : 'p-4']"
        >
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
                  <NButton
                    v-if="loading"
                    type="warning"
                    @click="handleStop"
                  >
                    <template #icon>
                      <SvgIcon icon="ri:stop-circle-line" />
                    </template>
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
          <div
            v-if="imageUploadFileKeysRef.length > 0"
            class="flex flex-wrap items-center gap-2 mb-2"
          >
            <div
              v-for="(key, index) in imageUploadFileKeysRef"
              :key="`img-${index}`"
              class="flex items-center px-2 py-1 text-xs bg-gray-100 rounded dark:bg-neutral-700"
            >
              <SvgIcon icon="ri:image-line" class="mr-1" />
              <span class="truncate max-w-[150px]">{{ key }}</span>
              <button
                class="ml-1 text-red-500 hover:text-red-700"
                @click="handleDeleteImageFile(index)"
              >
                <SvgIcon icon="ri:close-line" />
              </button>
            </div>
          </div>

          <div
            v-if="textUploadFileKeysRef.length > 0"
            class="flex flex-wrap items-center gap-2 mb-2"
          >
            <div
              v-for="(key, index) in textUploadFileKeysRef"
              :key="`txt-${index}`"
              class="flex items-center px-2 py-1 text-xs bg-gray-100 rounded dark:bg-neutral-700"
            >
              <SvgIcon icon="ri:file-text-line" class="mr-1" />
              <span class="truncate max-w-[150px]">{{ key }}</span>
              <button
                class="ml-1 text-red-500 hover:text-red-700"
                @click="handleDeleteTextFile(index)"
              >
                <SvgIcon icon="ri:close-line" />
              </button>
            </div>
          </div>
    
          <div class="flex items-center space-x-2">
            <div>
              <NUpload
                action="/api/upload-image"
                :headers="uploadHeaders"
                :show-file-list="false"
                response-type="json"
                accept="image/*"
                @finish="handleFinishImage"
                @before-upload="handleBeforeUpload"
              >
                <div
                  class="flex items-center justify-center h-10 px-2 transition rounded-md hover:bg-neutral-100 dark:hover:bg-[#414755] cursor-pointer"
                >
                  <span class="text-xl text-[#4f555e] dark:text-white">
                    <SvgIcon icon="ri:image-add-line" />
                  </span>
                </div>
              </NUpload>
            </div>

            <div>
              <NUpload
                action="/api/upload-image"
                :headers="uploadHeaders"
                :show-file-list="false"
                response-type="json"
                accept=".txt,.md,.json,.csv,.js,.ts,.py,.java,.html,.css,.xml,.yml,.yaml,.log,.ini,.config"
                @finish="handleFinishText"
                @before-upload="handleBeforeUpload"
              >
                <div
                  class="flex items-center justify-center h-10 px-2 transition rounded-md hover:bg-neutral-100 dark:hover:bg-[#414755] cursor-pointer"
                >
                  <span class="text-xl text-[#4f555e] dark:text-white">
                    <SvgIcon icon="ri:attachment-2" />
                  </span>
                </div>
              </NUpload>
            </div>

            <HoverButton @click="handleClear">
              <span class="text-xl text-[#4f555e] dark:text-white">
                <SvgIcon icon="ri:delete-bin-line" />
              </span>
            </HoverButton>

            <HoverButton
              v-if="!isMobile"
              @click="handleExport"
            >
              <span class="text-xl text-[#4f555e] dark:text-white">
                <SvgIcon icon="ri:download-2-line" />
              </span>
            </HoverButton>

            <HoverButton
              v-if="!isMobile"
              @click="showPrompt = true"
            >
              <span class="text-xl text-[#4f555e] dark:text-white">
                <IconPrompt class="w-[20px] m-auto" />
              </span>
            </HoverButton>

            <HoverButton
              v-if="!isMobile"
              :tooltip="usingContext ? $t('chat.clickTurnOffContext') : $t('chat.clickTurnOnContext')"
              :class="{ 'text-[#4b9e5f]': usingContext, 'text-[#a8071a]': !usingContext }"
              @click="handleToggleUsingContext"
            >
              <span class="text-xl">
                <SvgIcon icon="ri:chat-history-line" />
              </span>
            </HoverButton>

            <NSelect
              style="width: 250px"
              :value="currentChatModel"
              :options="authStore.session?.chatModels"
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
            <NAutoComplete
              v-model:value="prompt"
              :options="searchOptions"
              :render-label="renderOption"
            >
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

            <NButton
              type="primary"
              :disabled="buttonDisabled"
              @click="handleSubmit"
            >
              <template #icon>
                <span class="dark:text-black">
                  <SvgIcon icon="ri:send-plane-fill" />
                </span>
              </template>
            </NButton>
          </div>
        </NSpace>
      </div>
    </footer>

    <Prompt
      v-if="showPrompt"
      v-model:roomId="uuid"
      v-model:visible="showPrompt"
    />
  </div>
</template>
