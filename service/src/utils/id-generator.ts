// service/src/utils/id-generator.ts

import { v4 as uuidv4 } from 'uuid'

/**
 * 生成 AI 回复消息的 ID
 * 格式: msg_yyyymmddhhmmss_16位随机字符
 * @returns {string} 新的消息 ID
 */
export function generateMessageId(): string {
  const timestamp = new Date().toISOString()
    .replace(/[-:T.Z]/g, '')
    .substring(0, 14) // 格式: YYYYMMDDHHMMSS

  const random = uuidv4()
    .replace(/-/g, '')
    .substring(0, 16) // 16 位随机十六进制字符

  return `msg_${timestamp}_${random}`
}

/**
 * 生成用户提问消息的 ID
 * 格式: prompt_yyyymmddhhmmss_16位随机字符
 * @returns {string} 新的提示 ID
 */
export function generatePromptId(): string {
  const timestamp = new Date().toISOString()
    .replace(/[-:T.Z]/g, '')
    .substring(0, 14) // 格式: YYYYMMDDHHMMSS

  const random = uuidv4()
    .replace(/-/g, '')
    .substring(0, 16) // 16 位随机十六进制字符

  return `prompt_${timestamp}_${random}`
}
