<script lang="ts" setup>
import { computed, onMounted, onUnmounted, onUpdated, ref, nextTick } from 'vue'
import { NImage, NImageGroup } from 'naive-ui'
import MarkdownIt from 'markdown-it'
import mdKatex from '@traptitech/markdown-it-katex'
import mila from 'markdown-it-link-attributes'
import hljs from 'highlight.js'
import { useBasicLayout } from '@/hooks/useBasicLayout'
import { t } from '@/locales'
import { copyToClip } from '@/utils/copy'
import { SvgIcon } from '@/components/common'

interface Props {
  inversion?: boolean
  error?: boolean
  text?: string
  images?: string[]
  loading?: boolean
  asRawText?: boolean
}

const props = defineProps<Props>()

const { isMobile } = useBasicLayout()

const textRef = ref<HTMLElement | null>(null)

// -------- Markdown 图片预览核心逻辑 --------
// 初始化给一个空字符串，不要让它为 null
const markdownPreviewUrl = ref<string>('')
const markdownPreviewRef = ref<any>(null)

function handleImageClickFromMarkdown(src: string) {
  // 1. 设置 URL
  markdownPreviewUrl.value = src
  
  // 2. 等待 DOM 更新 src
  nextTick(() => {
    const inst = markdownPreviewRef.value
    if (inst && inst.$el) {
      // 核心修复：直接模拟点击 DOM 节点
      // Naive UI 的 NImage 监听的是 onClick，所以直接触发 click() 即可
      inst.$el.click()
      
      // 双重保险：如果根节点点击无效，尝试点击内部的 img 元素
      const innerImg = inst.$el.querySelector('img')
      if (innerImg) {
        innerImg.click()
      }
    }
  })
}
// -----------------------------------------

// 捕获 Markdown 内容点击事件
function handleMarkdownClick(e: MouseEvent) {
  const target = e.target as HTMLElement | null
  if (!target) return

  // 只有点到 img 标签才处理
  if (target.tagName.toLowerCase() === 'img') {
    // 阻止默认行为（防止浏览器打开新标签页等）
    e.preventDefault()
    e.stopPropagation()
    
    const src = (target as HTMLImageElement).src
    if (src) {
      handleImageClickFromMarkdown(src)
    }
  }
}

function isImage(filename: string | undefined): boolean {
  if (!filename) return false
  return /\.(jpg|jpeg|png|gif|webp|bmp|svg)$/i.test(filename)
}

const imageList = computed(() => {
  return (props.images || []).filter(file => isImage(file))
})

const fileList = computed(() => {
  return (props.images || []).filter(file => !isImage(file))
})

const mdi = new MarkdownIt({
  html: false,
  linkify: true,
  highlight(code, language) {
    const validLang = !!(language && hljs.getLanguage(language))
    if (validLang) {
      const lang = language ?? ''
      return highlightBlock(hljs.highlight(code, { language: lang }).value, lang)
    }
    return highlightBlock(hljs.highlightAuto(code).value, '')
  },
})

mdi.use(mila, { attrs: { target: '_blank', rel: 'noopener' } })
mdi.use(mdKatex, { blockClass: 'katexmath-block rounded-md p-[10px]', errorColor: ' #cc0000' })

function replaceBracketsWithDollar(input: string): string {
  const regex = /\\$$(.*?)\\$$/gs
  return input.replace(regex, (_, content) => `$$${content}$$`)
}

function replaceBracketsWithDollar2(input: string): string {
  const regex = /\\\( (.*?) \\\)/g
  return input.replace(regex, (_, content) => `$${content}$`)
}

const wrapClass = computed(() => {
  return [
    'text-wrap',
    'min-w-[20px]',
    'rounded-md',
    isMobile.value ? 'p-2' : 'px-3 py-2',
    props.inversion ? 'bg-[#d2f9d1]' : 'bg-[#f4f6f8]',
    props.inversion ? 'dark:bg-[#a1dc95]' : 'dark:bg-[#1e1e20]',
    props.inversion ? 'message-request' : 'message-reply',
    { 'text-red-500': props.error },
  ]
})

const text = computed(() => {
  let value = props.text ?? ''
  if (!props.asRawText) {
    value = replaceBracketsWithDollar(value)
    value = replaceBracketsWithDollar2(value)
    return mdi.render(value)
  }
  return value
})

function highlightBlock(str: string, lang?: string) {
  return `<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">${lang}</span><span class="code-block-header__copy">${t('chat.copyCode')}</span></div><code class="hljs code-block-body ${lang}">${str}</code></pre>`
}

function addCopyEvents() {
  if (textRef.value) {
    const copyBtnList = textRef.value.querySelectorAll('.code-block-header__copy')
    copyBtnList.forEach((btn) => {
      const handler = async () => {
        const code = btn.parentElement?.nextElementSibling?.textContent
        if (code) {
          await copyToClip(code)
          const originalText = t('chat.copyCode')
          btn.textContent = '复制成功'
          setTimeout(() => {
            btn.textContent = originalText
          }, 1000)
        }
      }
      btn.removeEventListener('click', handler)
      btn.addEventListener('click', handler)
    })
  }
}

function removeCopyEvents() {
  if (textRef.value) {
    const copyBtnList = textRef.value.querySelectorAll('.code-block-header__copy')
    copyBtnList.forEach((btn) => {
      btn.replaceWith(btn.cloneNode(true))
    })
  }
}

onMounted(() => {
  addCopyEvents()
})

onUpdated(() => {
  addCopyEvents()
})

onUnmounted(() => {
  removeCopyEvents()
})
</script>

<template>
  <div class="text-black" :class="wrapClass">
    <div
      ref="textRef"
      class="leading-relaxed break-words"
      @click="handleMarkdownClick"
    >
      <!-- 文本 / Markdown 渲染区域 -->
      <div v-if="!inversion" class="flex items-end">
        <div
          v-if="!asRawText"
          class="w-full markdown-body"
          :class="{ 'markdown-body-generate': loading }"
          v-html="text"
        />
        <div v-else class="w-full whitespace-pre-wrap" v-text="text" />
      </div>
      <div v-else class="whitespace-pre-wrap" v-text="text" />

      <!-- 附件图片区域 -->
      <div v-if="imageList.length > 0" class="flex flex-col gap-2 my-2">
        <NImageGroup>
          <NImage
            v-for="(v, i) of imageList"
            :key="`img-${i}`"
            :src="`/uploads/${v}`"
            alt="image"
            object-fit="contain"
            class="rounded-md shadow-sm cursor-pointer hover:opacity-90 excludeFastDel"
            :img-props="{
              style: { maxWidth: '100%', maxHeight: '300px', objectFit: 'contain' },
              alt: 'image',
            }"
          />
        </NImageGroup>
      </div>

      <!-- 非图片附件区域 -->
      <div v-if="fileList.length > 0" class="flex flex-col gap-2 my-2">
        <a
          v-for="(v, i) of fileList"
          :key="`file-${i}`"
          :href="`/uploads/${v}`"
          target="_blank"
          download
          class="flex items-center p-2 transition-colors bg-white border rounded-md shadow-sm dark:bg-neutral-800 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-700 group excludeFastDel"
          style="text-decoration: none; color: inherit;"
          @click.stop
        >
          <div class="flex items-center justify-center w-8 h-8 mr-2 bg-gray-100 rounded-full dark:bg-gray-700 text-gray-500 dark:text-gray-300">
            <SvgIcon icon="ri:file-text-line" class="text-lg" />
          </div>
          <div class="flex flex-col overflow-hidden">
            <span class="text-sm font-medium truncate w-48">{{ v }}</span>
            <span class="text-xs text-gray-400 group-hover:text-blue-500">点击下载</span>
          </div>
        </a>
      </div>

      <!-- 
        隐藏的 NImage 代理 
        1. 去掉了 v-if，保证组件一直存在，避免挂载延迟。
        2. 使用 fixed + 极大负值隐藏，确保不影响布局但可被 JS 触发。
      -->
      <div style="position: fixed; left: -99999px; top: -99999px; opacity: 0; pointer-events: none;">
        <NImage
          ref="markdownPreviewRef"
          :src="markdownPreviewUrl"
          :preview-disabled="false"
        />
      </div>
    </div>
  </div>
</template>

<style lang="less">
@import url(./style.less);
</style>
