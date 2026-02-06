<script setup lang='ts'>
import { onMounted, ref } from 'vue'
import { NButton, NInput, NInputNumber, NSelect, NSpin, NSwitch, useMessage } from 'naive-ui'
import type { ConfigState } from './model'
import { SiteConfig } from './model'
import { fetchChatConfig, fetchUpdateSite } from '@/api'
import { t } from '@/locales'

const ms = useMessage()
const loading = ref(false)
const saving = ref(false)
const config = ref(new SiteConfig())

async function fetchConfig() {
  try {
    loading.value = true
    const { data } = await fetchChatConfig<ConfigState>()
    config.value = data.siteConfig ? data.siteConfig : new SiteConfig()
  }
  finally {
    loading.value = false
  }
}

async function updateSiteInfo(site?: SiteConfig) {
  if (!site) return
  saving.value = true
  try {
    const { data } = await fetchUpdateSite(site)
    config.value = data
    ms.success(t('common.success'))
  }
  catch (error: any) {
    ms.error(error.message)
  }
  saving.value = false
}

onMounted(() => {
  fetchConfig()
})
</script>

<template>
  <NSpin :show="loading">
    <div class="p-4 space-y-5 min-h-[200px]">
      <div class="space-y-6">
        <!-- ===== 基本设置 ===== -->
        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.siteTitle') }}</span>
          <div class="flex-1">
            <NInput :value="config && config.siteTitle" placeholder="" @input="(val) => { if (config) config.siteTitle = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.siteDomain') }}</span>
          <div class="flex-1">
            <NInput :value="config && config.siteDomain" placeholder="" @input="(val) => { if (config) config.siteDomain = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.loginEnabled') }}</span>
          <div class="flex-1">
            <NSwitch :round="false" :disabled="config && config.loginEnabled" :value="config && config.loginEnabled" @update:value="(val) => { if (config) config.loginEnabled = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.loginSalt') }}</span>
          <div class="flex-1">
            <NInput :value="config && config.loginSalt" :placeholder="$t('setting.loginSaltTip')" @input="(val) => { if (config) config.loginSalt = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.registerEnabled') }}</span>
          <div class="flex-1">
            <NSwitch :round="false" :value="config && config.registerEnabled" @update:value="(val) => { if (config) config.registerEnabled = val }" />
          </div>
        </div>

        <div v-show="config && config.registerEnabled" class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.registerReview') }}</span>
          <div class="flex-1">
            <NSwitch :round="false" :value="config && config.registerReview" @update:value="(val) => { if (config) config.registerReview = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.registerMails') }}</span>
          <div class="flex-1">
            <NInput :value="config && config.registerMails" :placeholder="$t('setting.registerReviewTip')" @input="(val) => { if (config) config.registerMails = val }" />
          </div>
        </div>

        <!-- ===== 模型配置 ===== -->
        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.chatModels') }}</span>
          <div class="flex-1">
            <NInput :value="config && config.chatModels" placeholder="英文逗号分割 | English comma separated" type="textarea" :autosize="{ minRows: 1, maxRows: 4 }" @input="(val) => { if (config) config.chatModels = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">图片/非流式模型</span>
          <div class="flex-1">
            <NInput :value="config && config.imageModels" placeholder="英文逗号分割，如：gemini-3-pro-image, dall-e-3" type="textarea" :autosize="{ minRows: 1, maxRows: 4 }" @input="(val) => { if (config) config.imageModels = val }" />
          </div>
        </div>

        <!-- ===== 推理配置 ===== -->
        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">推理模型列表</span>
          <div class="flex-1">
            <NInput :value="config && config.reasoningModels" placeholder="英文逗号分隔，如：o3-mini,gpt-5.1" type="textarea" :autosize="{ minRows: 1, maxRows: 4 }" @input="(val) => { if (config) config.reasoningModels = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">reasoning_effort</span>
          <div class="flex-1">
            <NSelect
              :value="config && config.reasoningEffort"
              :options="[
                { label: 'none（不传参数）', key: 'none', value: 'none' },
                { label: 'low', key: 'low', value: 'low' },
                { label: 'medium', key: 'medium', value: 'medium' },
                { label: 'high', key: 'high', value: 'high' },
                { label: 'xhigh', key: 'xhigh', value: 'xhigh' },
              ]"
              @update-value="(val) => { if (config) config.reasoningEffort = val as any }"
            />
          </div>
        </div>

        <!-- ===== ✅ 联网搜索配置 ===== -->
        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">联网搜索启用</span>
          <div class="flex-1">
            <NSwitch :round="false" :value="config && config.webSearchEnabled" @update:value="(val) => { if (config) config.webSearchEnabled = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">搜索提供方</span>
          <div class="flex-1">
            <NSelect
              :value="config && config.webSearchProvider"
              :options="[
                { label: 'SearXNG', key: 'searxng', value: 'searxng' },
                { label: 'Tavily', key: 'tavily', value: 'tavily' },
              ]"
              @update-value="(val) => { if (config) config.webSearchProvider = val as any }"
            />
          </div>
        </div>

        <div v-if="config && config.webSearchProvider === 'searxng'" class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">SearXNG URL</span>
          <div class="flex-1">
            <NInput :value="config && config.searxngApiUrl" placeholder="例如：http://localhost:8080" @input="(val) => { if (config) config.searxngApiUrl = val }" />
          </div>
        </div>

        <div v-if="config && config.webSearchProvider === 'tavily'" class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">Tavily API Key</span>
          <div class="flex-1">
            <NInput
              :value="config && config.tavilyApiKey"
              type="password"
              show-password-on="click"
              placeholder="Tavily API Key（也可通过 env TAVILY_API_KEY 设置）"
              @input="(val) => { if (config) config.tavilyApiKey = val }"
            />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">每轮结果数</span>
          <div class="flex-1">
            <NInputNumber v-model:value="config.webSearchMaxResults" :min="1" :max="10" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">最大搜索轮数</span>
          <div class="flex-1">
            <NInputNumber v-model:value="config.webSearchMaxRounds" :min="1" :max="6" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">Planner 模型</span>
          <div class="flex-1">
            <NInput :value="config && config.webSearchPlannerModel" placeholder="可空；如 gpt-4o-mini。为空则用当前对话模型。" @input="(val) => { if (config) config.webSearchPlannerModel = val }" />
          </div>
        </div>

        <!-- ===== 其它开关 ===== -->
        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.globalAmount') }}</span>
          <div class="flex-1">
            <NInputNumber v-model:value="config.globalAmount" placeholder="" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.usageCountLimit') }}</span>
          <div class="flex-1">
            <NSwitch :round="false" :value="config && config.usageCountLimit" @update:value="(val) => { if (config) config.usageCountLimit = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.showWatermark') }}</span>
          <div class="flex-1">
            <NSwitch :round="false" :value="config && config.showWatermark" @update:value="(val) => { if (config) config.showWatermark = val }" />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]" />
          <NButton :loading="saving" type="primary" @click="updateSiteInfo(config)">
            {{ $t('common.save') }}
          </NButton>
        </div>
      </div>
    </div>
  </NSpin>
</template>
