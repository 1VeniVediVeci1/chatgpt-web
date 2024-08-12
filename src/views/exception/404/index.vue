<script lang="ts" setup>
import { NButton } from 'naive-ui'
import { useRouter } from 'vue-router'

const router = useRouter()

function goHome() {
  router.push('/')
}
</script>

<template>
  <div class="flex h-full">
    <div class="px-4 m-auto space-y-4 text-center max-[400px]">
      <h1 class="text-4xl text-slate-800 dark:text-neutral-200">
        找不到页面
      </h1>
      <p class="text-base text-slate-500 dark:text-neutral-400">
        抱歉，我们找不到您要查找的页面。也许您输入了错误的 URL？请确保拼写正确。
      </p>
      <div class="flex items-center justify-center text-center">
        <div class="w-[300px]">
          <img src="../../../icons/404.svg" alt="404">
        </div>
      </div>
      <NButton type="primary" @click="goHome">
        返回首页
      </NButton>
    </div>
  </div>
</template>
