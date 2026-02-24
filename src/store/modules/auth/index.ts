import { defineStore } from 'pinia'
import jwt_decode from 'jwt-decode'
import type { UserInfo } from '../user/helper'
import { useChatStore } from '../chat'
import { useUserStore } from '../user'
import { getToken, removeToken, setToken } from './helper'
import { store } from '@/store/helper'
import { fetchLogout, fetchSession } from '@/api'
import { UserConfig } from '@/components/common/Setting/model'

interface SessionResponse {
  auth: boolean
  authProxyEnabled: boolean
  /**
   * ✅ 后端返回的“全局默认 provider”
   * - openai-compatible | google
   * 注意：实际调用走哪个 provider 由 KeyConfig.keyModel 决定
   */
  model: 'openai-compatible' | 'google'
  allowRegister: boolean
  title: string
  chatModels: {
    label: string
    key: string
    value: string
  }[]
  allChatModels: {
    label: string
    key: string
    value: string
  }[]
  geminiChatModels: string[]
  nonStreamChatModels: string[]
  usageCountLimit: boolean
  showWatermark: boolean
  /**
   * ✅ 后端管理员配置：是否启用联网搜索功能
   */
  webSearchEnabled: boolean
  userInfo: { name: string; description: string; avatar: string; userId: string; root: boolean; config: UserConfig }
}

export interface AuthState {
  token: string | undefined
  session: SessionResponse | null
}

export const useAuthStore = defineStore('auth-store', {
  state: (): AuthState => ({
    token: getToken(),
    session: null,
  }),

  actions: {
    async getSession() {
      try {
        const { data } = await fetchSession<SessionResponse>()
        this.session = { ...data }
        return Promise.resolve(data)
      }
      catch (error) {
        return Promise.reject(error)
      }
    },

    async setToken(token: string) {
      this.token = token
      const decoded = jwt_decode(token) as UserInfo
      const userStore = useUserStore()
      if (decoded.config === undefined || decoded.config === null) {
        decoded.config = new UserConfig()
        decoded.config.chatModel = 'gpt-3.5-turbo'
      }

      await userStore.updateUserInfo(false, {
        avatar: decoded.avatar,
        name: decoded.name,
        description: decoded.description,
        root: decoded.root,
        config: decoded.config,
      })
      setToken(token)
    },

    async removeToken() {
      this.token = undefined
      const userStore = useUserStore()
      userStore.resetUserInfo()
      const chatStore = useChatStore()
      await chatStore.clearLocalChat()
      removeToken()
      await fetchLogout()
    },
  },
})

export function useAuthStoreWithout() {
  return useAuthStore(store)
}
