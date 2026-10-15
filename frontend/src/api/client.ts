import { ApiError } from './errors.ts'
import { SseDecoder } from './sse.ts'
import type { AskRequest, AskResponse, StreamMeta } from '../types.ts'

export { ApiError, isAbortError } from './errors.ts'

const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, '') ?? 'http://127.0.0.1:8000'

export interface AskStreamHandlers {
  /** 检索完成、大模型还没吐第一个字。用来把「检索中」换成「生成中」 */
  onMeta?: (meta: StreamMeta) => void
  /** 一段正文增量，可能一次来好几个字，也可能一个字一个字来 */
  onDelta?: (text: string) => void
}

/**
 * 流式问答。成功时返回最终的 AskResponse（来自 done 事件），失败时抛错。
 *
 * 不用 EventSource：它只支持 GET，而 /api/qa/ask/stream 需要 POST + JSON 请求体。
 * 所以拿 fetch + ReadableStream 自己读字节，切帧的活交给 SseDecoder（那边单独有测试）。
 */
export async function askStream(
  payload: AskRequest,
  handlers: AskStreamHandlers = {},
  signal?: AbortSignal,
): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/api/qa/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })

  // 关键：必须在读流之前判状态码。检索阶段的错误（503 没配 key、400 过滤表达式非法）
  // 是正常的 JSON 响应，读 body 就能拿到对人友好的 detail，原样透出去给用户看
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }
  if (!response.body) {
    throw new ApiError('服务端没有返回响应体', response.status)
  }

  const reader = response.body.getReader()
  const decoder = new SseDecoder()
  let result: AskResponse | null = null

  const handle = (events: ReturnType<SseDecoder['push']>) => {
    for (const evt of events) {
      switch (evt.event) {
        case 'meta':
          handlers.onMeta?.(evt.data)
          break
        case 'delta':
          if (evt.data.text) handlers.onDelta?.(evt.data.text)
          break
        case 'done':
          result = evt.data
          break
        case 'error':
          // 流已经开始了，状态码是 200，后端只能把错因当事件发出来
          throw new ApiError(evt.data.detail)
        default:
          break
      }
    }
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    handle(decoder.push(value))
  }
  handle(decoder.flush())

  if (!result) {
    throw new ApiError('连接中断：没有收到完整的回答')
  }
  return result
}

/** 从非 2xx 响应里取 FastAPI 的 {"detail": "..."}；取不到就退回状态码文案 */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (body && typeof body.detail === 'string') return body.detail
    return JSON.stringify(body)
  } catch {
    return `请求失败（HTTP ${response.status}）`
  }
}
