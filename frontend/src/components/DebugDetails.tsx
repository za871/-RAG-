import type { AskResponse } from '../types'

interface Props {
  result: AskResponse
}

function usageSummary(usage: Record<string, unknown>): string {
  const keys = ['input_tokens', 'output_tokens', 'total_tokens']
  const parts = keys
    .filter((k) => typeof usage[k] === 'number')
    .map((k) => `${k}=${usage[k]}`)
  // usage 是原样透传的 dict，字段名由 provider 决定；认识的键排前面，剩下的整体兜底
  return parts.length > 0 ? parts.join(' / ') : JSON.stringify(usage)
}

/**
 * 折叠的调试信息。
 *
 * 只放「界面上解释不了、但排查时一定要看」的东西：真正发给大模型的两条消息原文、
 * token 用量、引用编号的校验结果。不做成独立页面——那是给调参用的，
 * 而这里的定位是「答案看起来不对时，就地能翻一眼」。
 */
export default function DebugDetails({ result }: Props) {
  const { debug } = result
  const hasUsage = Object.keys(debug.usage).length > 0

  return (
    <details className="debug">
      <summary>调试信息</summary>
      <dl className="debug-grid">
        <dt>模型 / 温度</dt>
        <dd>
          {debug.model} / {debug.temperature}
        </dd>
        <dt>是否调用了大模型</dt>
        <dd>
          {debug.llm_called ? '是' : '否（检索侧已拒答，省掉一次调用）'}
        </dd>
        <dt>检索状态</dt>
        <dd>
          {debug.retrieval_status}
          {debug.retrieval_reason ? ` —— ${debug.retrieval_reason}` : ''}
        </dd>
        <dt>证据字符数</dt>
        <dd>{debug.evidence_chars}</dd>
        <dt>引用的编号</dt>
        <dd>{debug.cited_indices.length > 0 ? debug.cited_indices.join(', ') : '（无）'}</dd>
        <dt>越界的编号</dt>
        <dd>
          {debug.invalid_citation_indices.length > 0
            ? `${debug.invalid_citation_indices.join(', ')}（模型编造的，已从正文剔除）`
            : '（无）'}
        </dd>
        <dt>模型自报证据不足</dt>
        <dd>{debug.llm_insufficient ? '是' : '否'}</dd>
        {hasUsage && (
          <>
            <dt>token 用量</dt>
            <dd>{usageSummary(debug.usage)}</dd>
          </>
        )}
      </dl>

      {debug.messages.length > 0 && (
        <details className="debug-messages">
          <summary>实际发给大模型的消息（{debug.messages.length} 条）</summary>
          {debug.messages.map((message, i) => (
            <div key={i} className="debug-message">
              <div className="debug-role">{message.role}</div>
              <pre>{message.content}</pre>
            </div>
          ))}
        </details>
      )}

      {debug.answer_raw && debug.answer_raw !== result.answer && (
        <details className="debug-messages">
          <summary>模型原始输出（角标剔除前）</summary>
          <pre>{debug.answer_raw}</pre>
        </details>
      )}
    </details>
  )
}
