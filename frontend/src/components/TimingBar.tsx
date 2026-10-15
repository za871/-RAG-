import type { AskResponse, StreamMeta } from '../types'

interface Props {
  result: AskResponse | null
  meta: StreamMeta | null
}

function ms(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`
}

/**
 * 耗时与检索概况。
 *
 * 这行不是装饰：冷启动的第一次问答要加载重排权重（~2.2G，实测 6.3 秒）+
 * langchain-openai 的首次初始化（实测 3.7 秒），总耗时 10s+ 是正常的。
 * 把 retrieval_ms / llm_ms 拆开显示，「为什么这么久」才有答案——
 * 是检索慢（冷启动/重排）还是模型慢（推理 token 多）。
 */
export default function TimingBar({ result, meta }: Props) {
  const timings = result?.timings
  const debug = result?.debug
  const evidenceCount = debug?.evidence_count ?? meta?.evidence_count ?? null
  const dropped = debug?.evidence_dropped ?? meta?.evidence_dropped ?? null
  const rerank = debug?.rerank ?? meta?.rerank ?? null
  const hitCount = debug?.hit_count ?? meta?.hit_count ?? null

  return (
    <div className="timing-bar">
      {timings ? (
        <>
          <span title="检索 + 重排的总耗时，重排算在里面">
            检索 <b>{ms(timings.retrieval_ms)}</b>
          </span>
          <span title="大模型调用耗时；拒答路径不调模型，恒为 0">
            生成 <b>{ms(timings.llm_ms)}</b>
          </span>
          <span title="端到端，含证据组装、文件名回查、引用解析">
            合计 <b>{ms(timings.total_ms)}</b>
          </span>
        </>
      ) : (
        <span>
          检索 <b>{ms(meta?.retrieval_ms ?? 0)}</b>
        </span>
      )}
      {hitCount !== null && <span>命中 {hitCount} 块</span>}
      {evidenceCount !== null && <span>证据 {evidenceCount} 条</span>}
      {dropped ? (
        <span className="timing-warn" title="超出 QA_MAX_CONTEXT_CHARS 预算被整条丢弃的证据数">
          丢弃 {dropped} 条
        </span>
      ) : null}
      {rerank !== null && (
        <span className={rerank ? 'timing-on' : 'timing-off'}>
          重排 {rerank ? '开' : '关'}
        </span>
      )}
      {result?.debug.model && <span className="timing-model">{result.debug.model}</span>}
    </div>
  )
}
