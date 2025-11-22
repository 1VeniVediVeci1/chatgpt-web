<script lang="ts" setup>
import { computed, onMounted, onUnmounted, onUpdated, ref } from 'vue'
import { NImage, NModal } from 'naive-ui'
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

const textRef = ref<HTMLElement>()

// --- 图片预览逻辑 (全端统一) ---
const previewSrc = ref<string>('')
const showPreview = ref(false)

function handleImageClick(src: string) {
  // 不再区分移动端或PC端，统一使用自定义弹窗预览
  previewSrc.value = src
  showPreview.value = true
}
// ---------------------------

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
    const copyBtn = textRef.value.querySelectorAll('.code-block-header__copy')
    copyBtn.forEach((btn) => {
      btn.addEventListener('click', () => {
        const code = btn.parentElement?.nextElementSibling?.textContent
        if (code) {
          copyToClip(code).then(() => {
            btn.textContent = '复制成功'
            setTimeout(() => {
              btn.textContent = '复制代码'
            }, 1000)
          })
        }
      })
    })
  }
}

function removeCopyEvents() {
  if (textRef.value) {
    const copyBtn = textRef.value.querySelectorAll('.code-block-header__copy')
    copyBtn.forEach((btn) => {
      btn.removeEventListener('click', () => { })
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
    <div ref="textRef" class="leading-relaxed break-words">
      <div v-if="!inversion" class="flex items-end">
        <div
          v-if="!asRawText" class="w-full markdown-body" :class="{ 'markdown-body-generate': loading }"
          v-html="text"
        />
        <div v-else class="w-full whitespace-pre-wrap" v-text="text" />
      </div>
      <div v-else class="whitespace-pre-wrap" v-text="text" />

      <!-- 渲染图片 -->
      <div v-if="imageList.length > 0" class="flex flex-col gap-2 my-2">
        <div
          v-for="(v, i) of imageList"
          :key="`img-${i}`"
          class="excludeFastDel"
          @click.stop="handleImageClick(`/uploads/${v}`)"
        >
          <!-- 
            class: cursor-pointer 提示用户可点击。
          -->
          <NImage
            :src="`/uploads/${v}`"
            alt="image"
            object-fit="contain"
            class="rounded-md shadow-sm cursor-pointer hover:opacity-90"
            :img-props="{ style: { maxWidth: '100%', maxHeight: '300px' }, alt: 'image' }"
          />
        </div>
      </div>

      <!-- 渲染非图片文件 -->
      <div v-if="fileList.length > 0" class="flex flex-col gap-2 my-2">
        <a
          v-for="(v, i) of fileList"
          :key="`file-${i}`"
          :href="`/uploads/${v}`"
          target="_blank"
          download
          class="flex items-center p-2 transition-colors bg-white border rounded-md shadow-sm dark:bg-neutral-800 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-700 group"
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

      <!-- 全端通用的图片预览弹窗 -->
      <NModal
        v-model:show="showPreview"
        preset="card"
        style="width: auto; max-width: 95vw; background-color: transparent; box-shadow: none; border: none;"
        :header-style="{ display: 'none' }"
        :content-style="{ padding: 0, display: 'flex', justifyContent: 'center', alignItems: 'center' }"
      >
        <div class="relative w-full flex justify-center items-center cursor-zoom-out" @click="showPreview = false">
          <img
            v-if="previewSrc"
            :src="previewSrc"
            class="max-w-[95vw] max-h-[90vh] object-contain rounded-md shadow-2xl bg-black/50 backdrop-blur-sm"
            @click.stop
          >
          <!-- 
            @click.stop on img: 防止点击图片本身关闭弹窗（如果希望点击背景关闭，点击图片不关闭）
            如果希望点击图片也关闭，可以去掉 @click.stop
          -->
        </div>
      </NModal>

    </div>
  </div>
</template>

<style lang="less">
@import url(./style.less);
</style>
