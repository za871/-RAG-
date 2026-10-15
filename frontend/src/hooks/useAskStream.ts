import { useCallback, useRef, useState } from 'react'

import { ApiError, askStream, isAbortError } from '../api/client'
import type { AskRequest, AskResponse, StreamMeta } from '../types'

/**
 * idle        还没有提问
 * retrieving  请求已发出，检索阶段（后端在跑 embedding + Milvus + 重排，界面拿不到任何字节）
 * streaming   已收到 meta，正文正在逐字到达
 * done        收到完整的 AskResponse
 * stopped     用户点了「停止生成」，正文只到一半
 * error       失败：可能是检索阶段的 4xx/5xx，也可能是流中途的 error 事件
 */
export type AskPhase = 'idle' | 'retrieving' | 'streaming' | 'done' | 'stopped' | 'error'

export interface AskState {
  phase: AskPhase
  /** 本次提问的问题。和输入框分开存，输入框改了不影响已提交的这轮 */
  query: string
  /** 流式累积的正文。done 之后会被替换成 result.answer（服务端已剔除越界角标） */
  text: string
  meta: StreamMeta | null
  result: AskResponse | null
  error: string | null
  /** error 是否来自后端（true 就显示后端原文案，false 只是通用提示） */
  errorFromServer: boolean
}

const IDLE: AskState = {
  phase: 'idle',
  query: '',
  text: '',
  meta: null,
  result: null,
  error: null,
  errorFromServer: false,
}

export function useAskStream() {
  const [state, setState] = useState<AskState>(IDLE)
  const abortRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setState(IDLE)
  }, [])

  const ask = useCallback(async (payload: AskRequest) => {
    // 上一轮还在跑就直接打断：两轮流同时往 text 里追加会拼出乱码
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setState({ ...IDLE, phase: 'retrieving', query: payload.query })

    try {
      const result = await askStream(
        payload,
        {
          onMeta: (meta) => {
            // meta 一到就说明检索完成了。此时到第一个 delta 之间还有一段空档
            //（冷启动时重排要加载权重、langchain 要首次初始化），界面靠这个状态区分
            //「在检索」和「在生成」，不至于全程只有一个转圈
            setState((s) => ({ ...s, meta, phase: 'streaming' }))
          },
          onDelta: (text) => {
            setState((s) => ({ ...s, phase: 'streaming', text: s.text + text }))
          },
        },
        controller.signal,
      )
      // 以 done 里的 answer 为准，不用累积的 text：服务端把越界的幻觉角标 [n] 剔掉了，
      // 累积的增量里还留着它们
      setState((s) => ({ ...s, phase: 'done', result, text: result.answer }))
    } catch (err) {
      if (isAbortError(err)) {
        // 用户主动停止不算错误，保留已经收到的正文，只是不再增长
        setState((s) => ({ ...s, phase: 'stopped' }))
        return
      }
      const fromServer = err instanceof ApiError
      setState((s) => ({
        ...s,
        phase: 'error',
        // 后端的 detail 是写给人看的（「仅支持PDF、DOCX、TXT文件」这种），原样显示；
        // 网络层的异常（Failed to fetch / CORS 被拦）文案对用户没意义，换成通用提示
        error: fromServer
          ? (err as Error).message
          : '无法连接到后端服务，请确认 uvicorn 已在 127.0.0.1:8000 启动。',
        errorFromServer: fromServer,
      }))
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }, [])

  const busy = state.phase === 'retrieving' || state.phase === 'streaming'

  return { state, ask, stop, reset, busy }
}
