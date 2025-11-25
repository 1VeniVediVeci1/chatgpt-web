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

// ---------- Markdown 图片预览逻辑 ----------
const markdownPreviewUrl = ref<string>('')
const markdownPreviewRef = ref<any>(null)

function handleImageClickFromMarkdown(src: string) {
  markdownPreviewUrl.value = src

  nextTick(() => {
    const nImageEl = markdownPreviewRef.value?.$el
    if (nImageEl) {
      const imgTag = nImageEl.querySelector('img')
      if (imgTag)
        imgTag.click()
      else
        nImageEl.click()
    }
  })
}

function handleMarkdownClick(e: MouseEvent) {
  const target = e.target as HTMLElement | null
  if (!target)
    return

  if (target.tagName.toLowerCase() === 'img') {
    // 避免点击隐藏预览图自身时死循环
    if (target.closest('.hidden-preview-image'))
      return

    e.stopPropagation()
    e.preventDefault()

    const src = (target as HTMLImageElement).src
    if (src)
      handleImageClickFromMarkdown(src)
  }
}
// ----------------------------------------

// 去掉 img:/txt: 前缀，兼容历史数据
function stripTypePrefix(filename: string): string {
  return filename.replace(/^(img:|txt:)/, '')
}

// 根据“真实文件名”判断是否为图片
function isImage(filename: string | undefined): boolean {
  if (!filename) return false
  const realName = stripTypePrefix(filename)
  return /\.(jpg|jpeg|png|gif|webp|bmp|svg)$/i.test(realName)
}

// 图片列表：只保留图片，并去掉前缀，直接返回真实文件名
const imageList = computed(() => {
  const list = props.images || []
  return list
    .filter(file => isImage(file))
    .map(file => stripTypePrefix(file))
})

// 文件列表：非图片附件，同样去掉前缀
const fileList = computed(() => {
  const list = props.images || []
  return list
    .filter(file => !isImage(file))
    .map(file => stripTypePrefix(file))
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
    <!-- 
      [修改说明]
      将 Markdown 文本渲染 div 与 附件显示 div 分开。
      click="handleMarkdownClick" 只绑定在文本区域，
      避免点击 NImageGroup 里的附件图片时，事件冒泡导致触发两次预览。
    -->
    
    <!-- 1. Markdown / 文本内容区域 -->
    <div
      ref="textRef"
      class="leading-relaxed break-words"
      @click="handleMarkdownClick"
    >
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
    </div>

    <!-- 2. 附件图片（移出 textRef div） -->
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

    <!-- 3. 附件文件（移出 textRef div） -->
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
          <span class="text-sm font-medium truncate max-w-[150px]">{{ v }}</span>
          <span class="text-xs text-gray-400 group-hover:text-blue-500">点击下载</span>
        </div>
      </a>
    </div>

    <!-- 用于 Markdown 内联图片预览的隐藏 NImage -->
    <NImage
      ref="markdownPreviewRef"
      :src="markdownPreviewUrl"
      :preview-disabled="false"
      class="hidden-preview-image"
      style="width: 0; height: 0; overflow: hidden; position: absolute; visibility: hidden;"
    />
  </div>
</template>

<style lang="less">
@import url(./style.less);
</style>
