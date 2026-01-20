// service/src/chatgpt/abortable.ts
export class AbortError extends Error {
  override name = 'AbortError'
  constructor(message = 'Aborted') {
    super(message)
  }
}

/**
 * 判断一个错误是否为“取消/中断”
 */
export function isAbortError(err: any, signal?: AbortSignal): boolean {
  if (signal?.aborted)
    return true

  const name = String(err?.name ?? '')
  const message = String(err?.message ?? '')
  const code = String(err?.code ?? '')

  return (
    name.includes('Abort')
    || code === 'ERR_CANCELED'
    || /aborted|canceled|cancelled/i.test(message)
  )
}

/**
 * 让任意 Promise 支持 AbortSignal：
 * - signal aborted 时立刻 reject AbortError
 * - promise 正常完成/失败时自动解绑监听
 */
export function abortablePromise<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal)
    return promise

  if (signal.aborted)
    return Promise.reject(new AbortError())

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      cleanup()
      reject(new AbortError())
    }

    const cleanup = () => {
      signal.removeEventListener('abort', onAbort)
    }

    signal.addEventListener('abort', onAbort, { once: true })

    promise.then(
      (v) => {
        cleanup()
        resolve(v)
      },
      (e) => {
        cleanup()
        reject(e)
      },
    )
  })
}
