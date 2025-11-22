<script lang="ts" setup>
import { computed, onMounted, onUnmounted, onUpdated, ref } from 'vue'
import { NImage, NImageGroup, NModal } from 'naive-ui'
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

// --- 图片预览逻辑 ---
const previewSrc = ref<string>('')
const showPreview = ref(false)

function handleImageClickFromMarkdown(src: string) {
  previewSrc.value = src
  showPreview.value = true
}
// ----------------------

function handleMarkdownClick(e: MouseEvent) {
  const target = e.target as HTMLElement | null
  if (!target)
    return

  if (target.tagName.toLowerCase() === 'img') {
    e.stopPropagation()
    const src = (target as HTMLImageElement).src
    if (src)
      handleImageClickFromMarkdown(src)
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
      // 先移除再绑定，避免重复绑定
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
      <!-- 文本 / Markdown -->
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

      <!-- 附件图片：使用 NImageGroup + NImage，直接使用 Naive UI 预览/缩放 -->
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

      <!-- 非图片附件：文件下载 -->
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

      <!-- Markdown 内 img 的预览：点击 Markdown 里的 <img>，弹出 Modal，里面用 NImageGroup + NImage -->
      <NModal
        v-model:show="showPreview"
        preset="card"
        style="width: auto; max-width: 95vw; background-color: transparent; box-shadow: none; border: none;"
        :header-style="{ display: 'none' }"
        :content-style="{ padding: 0, display: 'flex', justifyContent: 'center', alignItems: 'center' }"
      >
        <div class="relative w-full flex justify-center items-center cursor-zoom-out" @click="showPreview = false">
          <NImageGroup>
            <NImage
              v-if="previewSrc"
              :src="previewSrc"
              object-fit="contain"
              class="max-w-[95vw] max-h-[90vh] rounded-md shadow-2xl bg-black/50 backdrop-blur-sm cursor-zoom-in"
              :img-props="{ style: { maxWidth: '95vw', maxHeight: '90vh', objectFit: 'contain' } }"
              @click.stop
            />
          </NImageGroup>
        </div>
      </NModal>
    </div>
  </div>
</template>

<style lang="less">
@import url(./style.less);
</style>
