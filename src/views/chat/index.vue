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
let lastChatInfo: any = {}

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

const { uuid } = route.params as { uuid: string }

const currentChatHistory = computed(() => chatStore.getChatHistoryByCurrentActive)
const usingContext = computed(() => currentChatHistory?.value?.usingContext ?? true)
const dataSources = computed(() => chatStore.getChatByUuid(+uuid))
const conversationList = computed(() => dataSources.value.filter(item => (!item.inversion && !!item.conversationOptions)))

const prompt = ref<string>('')
const firstLoading = ref<boolean>(false)
const loading = ref<boolean>(false) // 控制 Stop 按钮显示
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

// 手动重置 loading 状态，防止卡死，但仅针对非当前生成任务
dataSources.value.forEach((item, index) => {
  if (item.loading)
    updateChatSome(+uuid, index, { loading: false })
})

function handleSubmit() {
  onConversation()
}

const imageUploadFileKeysRef = ref<string[]>([])
const textUploadFileKeysRef = ref<string[]>([])

// 轮询定时器
let pollInterval: NodeJS.Timeout | null = null

// ====== 状态检查：页面加载/房间切换时调用 ======
async function checkProcessStatus() {
  try {
    // 每次检查前先停止旧轮询，防止叠加
    stopPolling()
    
    const { data } = await fetchChatProcessStatus()
    
    // 只有当后端正在生成的任务属于当前房间时，才恢复 loading 状态
    if (data.isProcessing && data.roomId === +uuid) {
      loading.value = true
      
      // 恢复 lastChatInfo，确保 Stop 按钮点击时能传参
      lastChatInfo = {
        id: data.messageId,
        conversationId: null, // 恢复时 conversationId 可能未知，传 null 后端只更新文本
        text: '...',
      }

      // 检查 UI 上最后一条消息状态
      const lastIndex = dataSources.value.length - 1
      const lastItem = dataSources.value[lastIndex]
      
      // 如果当前页面没消息，或者最后一条是用户的提问 -> 补一个 AI 的占位气泡
      if (lastIndex < 0 || (lastItem && lastItem.inversion)) {
        addChat(
          +uuid,
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
      }
      else {
        // 如果已经有 AI 的气泡（可能是之前的 LOADING 状态被上面的 forEach 重置了），重新设为 true
        updateChatSome(+uuid, lastIndex, { loading: true })
      }

      // 开始轮询直到结束
      startPolling()
    }
  }
  catch (error) {
    console.error('checkProcessStatus error:', error)
  }
}

// ====== 轮询后端任务状态 ======
function startPolling() {
  stopPolling() // 安全起见
  pollInterval = setInterval(async () => {
    try {
      const { data } = await fetchChatProcessStatus()
      
      // 1. 如果当前房间切走了 (data.roomId !== +uuid)，虽然任务可能还在跑，但当前页面不需要 loading
      if (data.roomId !== +uuid) {
        stopPolling()
        loading.value = false
        return
      }

      // 2. 如果任务结束了 (!data.isProcessing) 且就在当前房间
      if (!data.isProcessing) {
        stopPolling()
        loading.value = false

        // 延迟拉取，确保后端数据库写入完成
        setTimeout(async () => {
          // [关键] 清空当前房间的本地缓存，强制 syncChat 发起网络请求拉取最新完整回复
          // 避免 syncChat 发现本地有数据直接 return，导致页面只显示空的占位符
          const chatIndex = chatStore.chat.findIndex(d => d.uuid === Number(uuid))
          if (chatIndex > -1)
            chatStore.chat[chatIndex].data = []

          await chatStore.syncChat(
            { uuid: Number(uuid) } as Chat.History,
            undefined, // lastId undefined 代表拉取最新一页
            () => {
              scrollToBottom()
            },
          )
        }, 1500)
      }
    }
    catch (e) {
      console.error('poll status error:', e)
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

// ================== 提问 / 重新回答 逻辑 ==================
async function onConversation() {
  let message = prompt.value

  if (loading.value)
    return

  if (!message || message.trim() === '')
    return

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
  addChat(
    +uuid,
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

  addChat(
    +uuid,
    {
      uuid: chatUuid,
      dateTime: new Date().toLocaleString(),
      text: '',
      loading: true,
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
        roomId: +uuid,
        uuid: chatUuid,
        prompt: message,
        uploadFileKeys,
        options,
        signal: controller.signal,
        onDownloadProgress: ({ event }) => {
          const xhr = event.target
          const { responseText } = xhr
          const lastIndex = responseText.lastIndexOf('\n', responseText.length - 2)
          let chunk = responseText
          if (lastIndex !== -1)
            chunk = responseText.substring(lastIndex)
          try {
            const data = JSON.parse(chunk)
            // 实时保存对话信息，供 Stop 功能使用
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
              +uuid,
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

            if (openLongReply && data.detail && data.detail.choices.length > 0 && data.detail.choices[0].finish_reason === 'length') {
              options.parentMessageId = data.id
              lastText = data.text
              message = ''
              return fetchChatAPIOnce()
            }

            scrollToBottomIfAtBottom()
          }
          catch (error) {
            //
          }
        },
      })
      updateChatSome(+uuid, dataSources.value.length - 1, { loading: false })
    }

    await fetchChatAPIOnce()
  }
  catch (error: any) {
    const errorMessage = error?.message ?? t('common.wrong')

    if (error.message === 'canceled') {
      updateChatSome(
        +uuid,
        dataSources.value.length - 1,
        {
          loading: false,
        },
      )
      scrollToBottomIfAtBottom()
      return
    }

    const currentChat = getChatByUuidAndIndex(+uuid, dataSources.value.length - 1)
    const iso1model = currentChatModel.value?.includes('o1')
    if (currentChat?.text && currentChat.text !== '' && !iso1model) {
      updateChatSome(
        +uuid,
        dataSources.value.length - 1,
        {
          text: `${currentChat.text}\n[${errorMessage}]`,
          error: false,
          loading: false,
        },
      )
      return
    }
    if (currentChat?.text && currentChat.text !== '' && iso1model) {
      updateChatSome(
        +uuid,
        dataSources.value.length - 1,
        {
          text: `${currentChat.text}`,
          error: false,
          loading: false,
        },
      )
      return
    }

    updateChat(
      +uuid,
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

async function onRegenerate(index: number) {
  if (loading.value)
    return

  controller = new AbortController()

  const { requestOptions } = dataSources.value[index]
  let responseCount = dataSources.value[index].responseCount || 1
  responseCount++

  let message = requestOptions?.prompt ?? ''

  let options: Chat.ConversationRequest = {}

  if (requestOptions.options)
    options = { ...requestOptions.options }

  let uploadFileKeys: string[] = []
  if (index > 0) {
    const previousMessage = dataSources.value[index - 1]
    if (previousMessage && previousMessage.inversion && previousMessage.images && previousMessage.images.length > 0)
      uploadFileKeys = previousMessage.images
  }

  loading.value = true
  const chatUuid = dataSources.value[index].uuid
  updateChat(
    +uuid,
    index,
    {
      dateTime: new Date().toLocaleString(),
      text: '',
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
        roomId: +uuid,
        uuid: chatUuid || Date.now(),
        regenerate: true,
        prompt: message,
        uploadFileKeys,
        options,
        signal: controller.signal,
        onDownloadProgress: ({ event }) => {
          const xhr = event.target
          const { responseText } = xhr
          const lastIndex = responseText.lastIndexOf('\n', responseText.length - 2)
          let chunk = responseText
          if (lastIndex !== -1)
            chunk = responseText.substring(lastIndex)
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
              +uuid,
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

            if (openLongReply && data.detail && data.detail.choices.length > 0 && data.detail.choices[0].finish_reason === 'length') {
              options.parentMessageId = data.id
              lastText = data.text
              message = ''
              return fetchChatAPIOnce()
            }
          }
          catch (error) {
            //
          }
        },
      })
      updateChatSome(+uuid, index, { loading: false })
    }
    await fetchChatAPIOnce()
  }
  catch (error: any) {
    if (error.message === 'canceled') {
      updateChatSome(
        +uuid,
        index,
        {
          loading: false,
        },
      )
      return
    }

    const errorMessage = error?.message ?? t('common.wrong')

    updateChat(
      +uuid,
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
  }
  finally {
    loading.value = false
  }
}

async function onResponseHistory(index: number, historyIndex: number) {
  const chat = (await fetchChatResponseoHistory(+uuid, dataSources.value[index].uuid || Date.now(), historyIndex)).data
  updateChat(
    +uuid,
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

// ================= 导出 / 删除 / 清空 / Stop =================
function handleExport() {
  if (loading.value)
    return

  const d = dialog.warning({
    title: t('chat.exportImage'),
    content: t('chat.exportImageConfirm'),
    positiveText: t('common.yes'),
    negativeText: t('common.no'),
    onPositiveClick: async () => {
      try {
        d.loading = true
        const ele = document.getElementById('image-wrapper')
        const canvas = await html2canvas(ele as HTMLDivElement, {
          useCORS: true,
        })
        const imgUrl = canvas.toDataURL('image/png')
        const tempLink = document.createElement('a')
        tempLink.style.display = 'none'
        tempLink.href = imgUrl
        tempLink.setAttribute('download', 'chat-shot.png')
        if (typeof tempLink.download === 'undefined')
          tempLink.setAttribute('target', '_blank')

        document.body.appendChild(tempLink)
        tempLink.click()
        document.body.removeChild(tempLink)
        window.URL.revokeObjectURL(imgUrl)
        d.loading = false
        ms.success(t('chat.exportSuccess'))
        Promise.resolve()
      }
      catch (error: any) {
        ms.error(t('chat.exportFailed'))
      }
      finally {
        d.loading = false
      }
    },
  })
}

function handleDelete(index: number, fast: boolean) {
  if (loading.value)
    return

  if (fast === true) {
    chatStore.deleteChatByUuid(+uuid, index)
  }
  else {
    dialog.warning({
      title: t('chat.deleteMessage'),
      content: t('chat.deleteMessageConfirm'),
      positiveText: t('common.yes'),
      negativeText: t('common.no'),
      onPositiveClick: () => {
        chatStore.deleteChatByUuid(+uuid, index)
      },
    })
  }
}

function updateCurrentNavIndex(index: number, newIndex: number) {
  currentNavIndexRef.value = newIndex
}

function handleClear() {
  if (loading.value)
    return

  dialog.warning({
    title: t('chat.clearChat'),
    content: t('chat.clearChatConfirm'),
    positiveText: t('common.yes'),
    negativeText: t('common.no'),
    onPositiveClick: () => {
      chatStore.clearChatByUuid(+uuid)
    },
  })
}

function handleEnter(event: KeyboardEvent) {
  if (!isMobile.value) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }
  else {
    if (event.key === 'Enter' && event.ctrlKey) {
      event.preventDefault()
      handleSubmit()
    }
  }
}

async function handleStop() {
  if (loading.value) {
    // 中断前端的可读流
    controller.abort()
    loading.value = false
    stopPolling()
    // 调用后端接口，真正停止后台线程并保存当前内容
    await fetchChatStopResponding(lastChatInfo.text || 'Stopped', lastChatInfo.id, lastChatInfo.conversationId)
  }
}

// ================ 上滚加载更多历史 ================
async function loadMoreMessage(event: any) {
  const chatIndex = chatStore.chat.findIndex(d => d.uuid === +uuid)
  if (chatIndex <= -1 || chatStore.chat[chatIndex].data.length <= 0)
    return

  const scrollPosition = event.target.scrollHeight - event.target.scrollTop

  const lastId = chatStore.chat[chatIndex].data[0].uuid
  await chatStore.syncChat({ uuid: +uuid } as Chat.History, lastId, () => {
    loadingms && loadingms.destroy()
    nextTick(() => scrollTo(event.target.scrollHeight - scrollPosition))
  }, () => {
    loadingms = ms.loading(
      '加载中...', {
        duration: 0,
      },
    )
  }, () => {
    allmsg && allmsg.destroy()
    allmsg = ms.warning('没有更多了', {
      duration: 1000,
    })
  })
}

const handleLoadMoreMessage = debounce(loadMoreMessage, 300)
const handleSyncChat
  = debounce(() => {
    chatStore.syncChat({ uuid: Number(uuid) } as Chat.History, undefined, () => {
      firstLoading.value = false
      const scrollRefDom = document.querySelector('#scrollRef')
      if (scrollRefDom)
        nextTick(() => scrollRefDom.scrollTop = scrollRefDom.scrollHeight)
      if (inputRef.value && !isMobile.value)
        inputRef.value?.focus()
    })
  }, 200)

async function handleScroll(event: any) {
  const scrollTop = event.target.scrollTop
  if (scrollTop < 50 && (scrollTop < prevScrollTop || prevScrollTop === undefined))
    handleLoadMoreMessage(event)
  prevScrollTop = scrollTop
}

// ================ 上下文开关 ================
async function handleToggleUsingContext() {
  if (!currentChatHistory.value)
    return

  currentChatHistory.value.usingContext = !currentChatHistory.value.usingContext
  chatStore.setUsingContext(currentChatHistory.value.usingContext, +uuid)
  if (currentChatHistory.value.usingContext)
    ms.success(t('chat.turnOnContext'))
  else
    ms.warning(t('chat.turnOffContext'))
}

// ================ Prompt 模板 & 文本提示 ================
const searchOptions = computed(() => {
  if (prompt.value.startsWith('/')) {
    return promptTemplate.value.filter((item: { title: string }) => item.title.toLowerCase().includes(prompt.value.substring(1).toLowerCase())).map((obj: { value: any }) => {
      return {
        label: obj.value,
        value: obj.value,
      }
    })
  }
  else {
    return []
  }
})

function renderOption(option: { label: string }) {
  for (const i of promptTemplate.value) {
    if (i.value === option.label)
      return [i.title]
  }
  return []
}

const placeholder = computed(() => {
  if (isMobile.value)
    return t('chat.placeholderMobile')
  return t('chat.placeholder')
})

const buttonDisabled = computed(() => {
  return loading.value || !prompt.value || prompt.value.trim() === ''
})

const footerClass = computed(() => {
  let classes = ['p-4']
  if (isMobile.value)
    classes = ['sticky', 'left-0', 'bottom-0', 'right-0', 'p-2', 'pr-3', 'overflow-hidden']
  return classes
})

async function handleSyncChatModel(chatModel: string) {
  nowSelectChatModel.value = chatModel
  await chatStore.setChatModel(chatModel, +uuid)
}

function formatTooltip(value: number) {
  return `${t('setting.maxContextCount')}: ${value}`
}

// ================ 附件上传 ================
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
  return {
    Authorization: `Bearer ${token}`,
  }
})

// ================ 生命周期 & 监听 ================
onMounted(() => {
  firstLoading.value = true
  handleSyncChat()
  // 进入房间时检查是否有正在进行的任务
  checkProcessStatus()

  if (authStore.token) {
    const chatModels = authStore.session?.chatModels
    if (chatModels != null && chatModels.filter(d => d.value === userStore.userInfo.config.chatModel).length <= 0)
      ms.error('你选择的模型已不存在，请重新选择 | The selected model not exists, please choose again.', { duration: 7000 })
  }
})

watch(() => chatStore.active, () => {
  handleSyncChat()
  // 切换房间时停止旧的轮询，并检查新房间的状态
  stopPolling()
  checkProcessStatus()
})

onUnmounted(() => {
  // 修改核心点：卸载组件（切房/刷新）时，不要 abort controller。
  // 否则会导致前端主动发终止信号，后端收到后会杀死生成线程，导致切回来时任务已断。
  /* if (loading.value) controller.abort() */
  
  stopPolling()
})
</script>

<template>
  <div class="flex flex-col w-full h-full">
    <HeaderComponent
      v-if="isMobile"
      :using-context="usingContext"
      :show-prompt="showPrompt"
      @export="handleExport" @toggle-using-context="handleToggleUsingContext"
      @toggle-show-prompt="showPrompt = true"
    />
    <main class="flex-1 overflow-hidden">
      <div id="scrollRef" ref="scrollRef" class="h-full overflow-hidden overflow-y-auto" @scroll="handleScroll">
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
                  <NButton v-if="loading" type="warning" @click="handleStop">
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
          <div v-if="imageUploadFileKeysRef.length > 0" class="flex flex-wrap items-center gap-2 mb-2">
            <div
              v-for="(key, index) in imageUploadFileKeysRef"
              :key="`img-${index}`"
              class="flex items-center px-2 py-1 text-xs bg-gray-100 rounded dark:bg-neutral-700"
            >
              <SvgIcon icon="ri:image-line" class="mr-1" />
              <span class="truncate max-w-[150px]">{{ key }}</span>
              <button class="ml-1 text-red-500 hover:text-red-700" @click="handleDeleteImageFile(index)">
                <SvgIcon icon="ri:close-line" />
              </button>
            </div>
          </div>

          <div v-if="textUploadFileKeysRef.length > 0" class="flex flex-wrap items-center gap-2 mb-2">
            <div
              v-for="(key, index) in textUploadFileKeysRef"
              :key="`txt-${index}`"
              class="flex items-center px-2 py-1 text-xs bg-gray-100 rounded dark:bg-neutral-700"
            >
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
                :headers="uploadHeaders"
                :show-file-list="false"
                response-type="json"
                accept="image/*"
                @finish="handleFinishImage"
                @before-upload="handleBeforeUpload"
              >
                <div class="flex items-center justify-center h-10 px-2 transition rounded-md hover:bg-neutral-100 dark:hover:bg-[#414755] cursor-pointer">
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
                <div class="flex items-center justify-center h-10 px-2 transition rounded-md hover:bg-neutral-100 dark:hover:bg-[#414755] cursor-pointer">
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
            <HoverButton v-if="!isMobile" @click="handleExport">
              <span class="text-xl text-[#4f555e] dark:text-white">
                <SvgIcon icon="ri:download-2-line" />
              </span>
            </HoverButton>
            <HoverButton v-if="!isMobile" @click="showPrompt = true">
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
            <NButton type="primary" :disabled="buttonDisabled" @click="handleSubmit">
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
    <Prompt v-if="showPrompt" v-model:roomId="uuid" v-model:visible="showPrompt" />
  </div>
</template>
