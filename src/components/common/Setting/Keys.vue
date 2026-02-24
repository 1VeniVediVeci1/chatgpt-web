<script lang="ts" setup>
import type { DataTableColumns } from 'naive-ui'
import { h, onMounted, reactive, ref } from 'vue'
import {
  NButton,
  NDataTable,
  NInput,
  NInputNumber,
  NModal,
  NSelect,
  NSpace,
  NSwitch,
  NTag,
  NTooltip,
  useDialog,
  useMessage,
} from 'naive-ui'
import { KeyConfig, Status, UserRole, apiModelOptions, userRoleOptions } from './model'
import { fetchGetKeys, fetchUpdateApiKeyStatus, fetchUpsertApiKey } from '@/api'
import { t } from '@/locales'
import { useAuthStore } from '@/store'
import { useBasicLayout } from '@/hooks/useBasicLayout'

const ms = useMessage()
const dialog = useDialog()
const authStore = useAuthStore()
const { isMobile } = useBasicLayout()

const loading = ref(false)
const show = ref(false)
const handleSaving = ref(false)
const keyConfig = ref(new KeyConfig('', 'openai-compatible', [], [], ''))

// 只保留 Token Policy（不再提供备注栏）
const editingMaxTokens = ref<number | null>(null)
const editingMinTokens = ref<number | null>(null)

const TOKEN_POLICY_REGEX = {
  max: new RegExp('MAX_TOKENS?\\s*[:=]\\s*(\\d+)', 'i'),
  min: new RegExp('MIN_TOKENS?\\s*[:=]\\s*(\\d+)', 'i'),
}

function parseTokenPolicy(remark: string) {
  const r = remark || ''
  const max = r.match(TOKEN_POLICY_REGEX.max)?.[1]
  const min = r.match(TOKEN_POLICY_REGEX.min)?.[1]
  return {
    max: max ? Number.parseInt(max, 10) : null,
    min: min ? Number.parseInt(min, 10) : null,
  }
}

// 注意：这里会把 remark 覆盖成仅用于策略存储的内容（不再保存“备注文本”）
function buildTokenPolicy(max: number | null, min: number | null) {
  const parts: string[] = []
  if (max !== null) parts.push(`[MAX_TOKENS:${max}]`)
  if (min !== null) parts.push(`[MIN_TOKENS:${min}]`)
  return parts.join(' ')
}

const keys = ref<any[]>([])

function createColumns(): DataTableColumns {
  return [
    {
      title: 'Key',
      key: 'key',
      resizable: true,
      width: 120,
      minWidth: 50,
      maxWidth: 120,
      ellipsis: true,
    },
    {
      title: 'Provider',
      key: 'keyModel',
      width: 160,
    },
    {
      title: 'Base url',
      key: 'baseUrl',
      width: 220,
    },
    {
      title: 'Chat Model',
      key: 'chatModels',
      width: 300,
      render(row: any) {
        return row.chatModels.map((chatModel: string) =>
          h(
            NTag,
            {
              style: { marginRight: '6px' },
              type: 'info',
              bordered: false,
            },
            { default: () => chatModel },
          ),
        )
      },
    },
    {
      title: 'User Roles',
      key: 'userRoles',
      width: 180,
      render(row: any) {
        return row.userRoles.map((userRole: UserRole) =>
          h(
            NTag,
            {
              style: { marginRight: '6px' },
              type: 'info',
              bordered: false,
            },
            { default: () => UserRole[userRole] },
          ),
        )
      },
    },
    {
      title: 'Token Policy',
      key: 'tokenPolicy',
      width: 220,
      render(row: any) {
        const { max, min } = parseTokenPolicy(row.remark)
        const nodes: any[] = []

        if (max !== null) {
          nodes.push(
            h(
              NTag,
              { type: 'warning', bordered: false, size: 'small', style: { marginRight: '4px' } },
              { default: () => `≤ ${max} Tokens` },
            ),
          )
        }
        if (min !== null) {
          nodes.push(
            h(NTag, { type: 'error', bordered: false, size: 'small' }, { default: () => `> ${min} Tokens` }),
          )
        }

        return nodes.length ? nodes : h('span', { style: { color: '#999' } }, '-')
      },
    },
    {
      title: 'Action',
      key: '_id',
      width: 220,
      fixed: 'right',
      render(row: any) {
        const actions: any[] = []
        actions.push(
          h(
            NButton,
            {
              size: 'small',
              style: { marginRight: '6px' },
              type: 'error',
              onClick: () => handleUpdateApiKeyStatus(row._id as string, Status.Disabled),
            },
            { default: () => t('common.delete') },
          ),
        )
        if (row.status === Status.Normal) {
          actions.push(
            h(
              NButton,
              {
                size: 'small',
                style: { marginRight: '6px' },
                type: 'info',
                onClick: () => handleEditKey(row as KeyConfig),
              },
              { default: () => t('common.edit') },
            ),
          )
        }
        return actions
      },
    },
  ]
}

const columns = createColumns()

const pagination = reactive({
  page: 1,
  pageSize: 100,
  pageCount: 1,
  itemCount: 1,
  prefix({ itemCount }: { itemCount: number | undefined }) {
    return `Total is ${itemCount}.`
  },
  showSizePicker: true,
  pageSizes: [100],
  onChange: (page: number) => {
    pagination.page = page
    handleGetKeys(pagination.page)
  },
  onUpdatePageSize: (pageSize: number) => {
    pagination.pageSize = pageSize
    pagination.page = 1
    handleGetKeys(pagination.page)
  },
})

async function handleGetKeys(page: number) {
  if (loading.value) return
  keys.value.length = 0
  loading.value = true

  const size = pagination.pageSize
  const data = (await fetchGetKeys(page, size)).data
  data.keys.forEach((key: any) => keys.value.push(key))

  keyConfig.value = keys.value[0]
  pagination.page = page
  pagination.pageCount = data.total / size + (data.total % size === 0 ? 0 : 1)
  pagination.itemCount = data.total

  loading.value = false
}

async function handleUpdateApiKeyStatus(id: string, status: Status) {
  dialog.warning({
    title: t('chat.deleteKey'),
    content: t('chat.deleteKeyConfirm'),
    positiveText: t('common.yes'),
    negativeText: t('common.no'),
    onPositiveClick: async () => {
      await fetchUpdateApiKeyStatus(id, status)
      ms.info('OK')
      await handleGetKeys(pagination.page)
    },
  })
}

async function handleUpdateKeyConfig() {
  if (!keyConfig.value.key) {
    ms.error('Api key is required')
    return
  }

  // remark 字段仅用于存储 Token Policy
  keyConfig.value.remark = buildTokenPolicy(editingMaxTokens.value, editingMinTokens.value)

  handleSaving.value = true
  try {
    await fetchUpsertApiKey(keyConfig.value)
    await handleGetKeys(pagination.page)
    show.value = false
  }
  catch (error: any) {
    ms.error(error.message)
  }
  handleSaving.value = false
}

function handleNewKey() {
  keyConfig.value = new KeyConfig('', 'openai-compatible', [], [], '')
  editingMaxTokens.value = null
  editingMinTokens.value = null
  show.value = true
}

function handleEditKey(key: KeyConfig) {
  keyConfig.value = { ...key }
  const { max, min } = parseTokenPolicy((key as any).remark)
  editingMaxTokens.value = max
  editingMinTokens.value = min
  show.value = true
}

onMounted(async () => {
  await handleGetKeys(pagination.page)
})
</script>

<template>
  <div class="p-4 space-y-5 min-h-[300px]">
    <div class="space-y-6">
      <NSpace vertical :size="12">
        <NSpace>
          <NButton @click="handleNewKey()">
            New Key
          </NButton>
        </NSpace>
        <NDataTable
          remote
          :loading="loading"
          :row-key="(rowData) => rowData._id"
          :columns="columns"
          :data="keys"
          :pagination="pagination"
          :max-height="444"
          :scroll-x="1300"
          striped
          @update:page="handleGetKeys"
        />
      </NSpace>
    </div>
  </div>

  <NModal v-model:show="show" :auto-focus="false" preset="card" :style="{ width: !isMobile ? '50%' : '100%' }">
    <div class="p-4 space-y-5 min-h-[200px]">
      <div class="space-y-6">
        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.apiModel') }}</span>
          <div class="flex-1">
            <NSelect
              style="width: 100%"
              :value="keyConfig.keyModel"
              :options="apiModelOptions"
              @update-value="value => keyConfig.keyModel = value"
            />
          </div>
          <p v-if="!isMobile">
            <a v-if="keyConfig.keyModel === 'openai-compatible'" target="_blank" href="https://platform.openai.com/account/api-keys">Get OpenAI API Key</a>
            <a v-else-if="keyConfig.keyModel === 'google'" target="_blank" href="https://aistudio.google.com/app/apikey">Get Gemini API Key</a>
          </p>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.api') }}</span>
          <div class="flex-1">
            <NInput
              v-model:value="keyConfig.key"
              type="textarea"
              :autosize="{ minRows: 3, maxRows: 4 }"
              placeholder=""
            />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.apiBaseUrl') }}</span>
          <div class="flex-1">
            <NInput
              v-model:value="keyConfig.baseUrl"
              style="width: 100%"
              placeholder="可选：该 Key 专用 baseUrl（优先级最高）。openai-compatible 填 OpenAI/兼容网关；google 填 Gemini 反代域名"
            />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.chatModels') }}</span>
          <div class="flex-1">
            <NSelect
              style="width: 100%"
              multiple
              :value="keyConfig.chatModels"
              :options="authStore.session?.allChatModels"
              @update-value="value => keyConfig.chatModels = value"
            />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.userRoles') }}</span>
          <div class="flex-1">
            <NSelect
              style="width: 100%"
              multiple
              :value="keyConfig.userRoles"
              :options="userRoleOptions"
              @update-value="value => keyConfig.userRoles = value"
            />
          </div>
        </div>

        <!-- Token Policy -->
        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">Token Policy</span>
          <div class="flex-1 flex gap-4">
            <div class="flex-1">
              <NTooltip trigger="hover">
                <template #trigger>
                  <NInputNumber v-model:value="editingMaxTokens" placeholder="Max" :min="0" :show-button="false">
                    <template #prefix>≤ </template>
                  </NInputNumber>
                </template>
                Max Input Tokens (Use this key if tokens ≤ val)
              </NTooltip>
            </div>
            <div class="flex-1">
              <NTooltip trigger="hover">
                <template #trigger>
                  <NInputNumber v-model:value="editingMinTokens" placeholder="Min" :min="0" :show-button="false">
                    <template #prefix>&gt; </template>
                  </NInputNumber>
                </template>
                Min Input Tokens (Use this key if tokens &gt; val)
              </NTooltip>
            </div>
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]">{{ $t('setting.status') }}</span>
          <div class="flex-1">
            <NSwitch
              :round="false"
              :value="keyConfig.status === Status.Normal"
              @update:value="(val) => { keyConfig.status = val ? Status.Normal : Status.Disabled }"
            />
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <span class="flex-shrink-0 w-[100px]" />
          <NButton type="primary" :loading="handleSaving" @click="handleUpdateKeyConfig()">
            {{ $t('common.save') }}
          </NButton>
        </div>
      </div>
    </div>
  </NModal>
</template>
