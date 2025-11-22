<script lang="ts" setup>
import { computed, onMounted, onUnmounted, onUpdated, ref } from 'vue'
import { NImage } from 'naive-ui' 
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
  
function isImage(filename: string | undefined): boolean {
  if (!filename) return false
  // 简单的后缀判断，可根据需要补充
  return /\.(jpg|jpeg|png|gif|webp|bmp|svg)$/i.test(filename)
}

// 计算属性：过滤出图片
const imageList = computed(() => {
  return (props.images || []).filter(file => isImage(file))
})

// 计算属性：过滤出非图片（文档等）
const fileList = computed(() => {
  return (props.images || []).filter(file => !isImage(file))
})
  
const props = defineProps<Props>()

const { isMobile } = useBasicLayout()

const textRef = ref<HTMLElement>()

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
    // 正则表达式解释：
    // \\[     匹配字符 '\['
    // (.*?)   非贪婪匹配任意字符，并捕获为分组 1
    // \\]     匹配字符 '\]'
    const regex = /\\\[(.*?)\\\]/gs;

    // 使用 replace 方法进行替换
    return input.replace(regex, (_, content) => `$$${content}$$`);
}

function replaceBracketsWithDollar2(input: string): string {
    // 正则表达式解释：
    // \\(     匹配字符 '\('
    // (.*?)   非贪婪匹配任意字符，并捕获为分组 1
    // \\)     匹配字符 '\)'
    const regex = /\\\( (.*?) \\\)/g;

    // 使用 replace 方法进行替换
    return input.replace(regex, (_, content) => `$${content}$`);
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
  let value = props.text ?? '';
  if (!props.asRawText) {
    value = replaceBracketsWithDollar(value);
    value = replaceBracketsWithDollar2(value);
    return mdi.render(value);
  }
  return value;
});

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
      
      <!-- 渲染图片：使用 NImage 替换 img -->
      <div v-if="imageList.length > 0" class="flex flex-col gap-2 my-2">
        <div 
          v-for="(v, i) of imageList" 
          :key="`img-${i}`" 
          @click.stop
        >
          <NImage
            :src="`/uploads/${v}`" 
            alt="image" 
            object-fit="contain"
            class="rounded-md shadow-sm cursor-pointer hover:opacity-90"
            :img-props="{ style: { maxWidth: '100%', width: '300px' }, alt: 'image' }"
          />
        </div>
      </div>

      <!-- 渲染非图片文件：添加 download 属性 -->
      <div v-if="fileList.length > 0" class="flex flex-col gap-2 my-2">
        <a 
          v-for="(v, i) of fileList" 
          :key="`file-${i}`" 
          :href="`/uploads/${v}`" 
          target="_blank"
          download 
          class="flex items-center p-2 transition-colors bg-white border rounded-md shadow-sm dark:bg-neutral-800 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-700 group"
          style="text-decoration: none; color: inherit;"
        >
          <div class="flex items-center justify-center w-8 h-8 mr-2 bg-gray-100 rounded-full dark:bg-gray-700 text-gray-500 dark:text-gray-300">
            <SvgIcon icon="ri:file-text-line" class="text-lg" />
          </div>
          <div class="flex flex-col overflow-hidden">
            <span class="text-sm font-medium truncate w-48">{{ v }}</span>
            <!-- 修改提示文字 -->
            <span class="text-xs text-gray-400 group-hover:text-blue-500">点击下载</span>
          </div>
        </a>
      </div>

    </div>
  </div>
</template>

<style lang="less">
@import url(./style.less);
</style>
