import type { AskResponse } from '../types'

interface Props {
  result: AskResponse
}

/**
 * 拒答 / 校验异常的状态条。
 *
 * 三种拒答的根因完全不同，排查方向也完全不同，所以必须分开呈现：
 *   no_evidence           检索召回本身就是空的（库空、被 topic/日期过滤干净、分词为空）
 *   refused               召回到了，但全部低于 MIN_EVIDENCE_SCORE（相关性不够）
 *   insufficient_evidence 召回到了、也够相关，是大模型自己判定证据不足以回答
 * 只显示后端的罐装文案会让这三种看起来一模一样，用户不知道该「换问法」还是「先传文档」。
 */
const COPY: Record<string, { icon: string; title: string; hint: string; tone: string }> = {
  no_evidence: {
    icon: '🗂',
    title: '知识库中没有找到相关材料',
    hint: '可以先上传相关政策文档，或把问题里的关键词写得更具体。',
    tone: 'info',
  },
  refused: {
    icon: '⚠',
    title: '检索到的材料与问题相关性不足',
    hint: '为避免给出不准确的回答已拒答。换个更贴近政策原文的问法通常能召回。',
    tone: 'warn',
  },
  insufficient_evidence: {
    icon: '⚠',
    title: '已检索到材料，但不足以回答该问题',
    hint: '材料里没有覆盖问题所需的细节。补充背景信息，或换一个更具体的问题。',
    tone: 'warn',
  },
}

export default function StatusNotice({ result }: Props) {
  // 引用校验失败是另一类异常：status=ok 说明模型认为有据可依，
  // 但一条引用都没标（或标了越界号）说明这段答案其实无据——这是最该被看见的情况
  if (result.status === 'ok' && !result.citations_valid) {
    return (
      <div className="notice notice-warn">
        <span className="notice-icon">⚠</span>
        <div>
          <strong>答案未通过引用校验</strong>
          <p>
            模型给出了结论，但{result.citations.length === 0 ? '没有标注任何引用' : '包含了越界的引用编号'}
            ，无法追溯到具体条款。以下内容仅供参考，请以原文为准。
          </p>
        </div>
      </div>
    )
  }

  if (result.status === 'ok') return null

  const copy = COPY[result.status] ?? COPY.no_evidence
  return (
    <div className={`notice notice-${copy.tone}`}>
      <span className="notice-icon">{copy.icon}</span>
      <div>
        <strong>{copy.title}</strong>
        <p>{result.answer}</p>
        {/* reason 是检索侧的机器可读原因（reason=...），比罐装文案更贴近实际触发条件 */}
        {result.reason && <p className="notice-reason">检索侧判定：{result.reason}</p>}
        <p className="notice-hint">{copy.hint}</p>
      </div>
    </div>
  )
}
