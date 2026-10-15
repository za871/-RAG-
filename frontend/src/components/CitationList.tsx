import type { Citation } from '../types'

interface Props {
  citations: Citation[]
  highlight: number | null
  onHover: (n: number | null) => void
}

function scoreLabel(score: number | null): string | null {
  if (score === null) return null
  // 重排分已经 sigmoid 归一化到 (0,1)，两位小数足够区分；界面不做「阈值线」那种暗示，
  // 因为 MIN_EVIDENCE_SCORE 只用于拒答判定，低于它的证据根本不会出现在这里
  return score.toFixed(2)
}

export default function CitationList({ citations, highlight, onHover }: Props) {
  if (citations.length === 0) return null

  return (
    <section className="citations">
      <h2 className="section-title">
        依据 <span className="count">{citations.length}</span>
      </h2>
      <ol className="citation-list">
        {citations.map((citation) => (
          <li
            key={citation.index}
            id={`citation-${citation.index}`}
            className={`citation-card${highlight === citation.index ? ' is-active' : ''}`}
            onMouseEnter={() => onHover(citation.index)}
            onMouseLeave={() => onHover(null)}
          >
            <div className="citation-head">
              <span className="citation-index">{citation.index}</span>
              <span className="citation-file" title={citation.filename}>
                {citation.filename || '（文件名缺失）'}
              </span>
              {scoreLabel(citation.rerank_score) !== null && (
                <span className="citation-score" title="cross-encoder 重排分，已归一化到 0~1">
                  相关度 {scoreLabel(citation.rerank_score)}
                </span>
              )}
            </div>
            <div className="citation-meta">
              {citation.section && <span>章节：{citation.section}</span>}
              {citation.page !== null && <span>第 {citation.page} 页</span>}
              {citation.topic && <span>主题：{citation.topic}</span>}
              {citation.effective_date && <span>生效日期：{citation.effective_date}</span>}
            </div>
            <p className="citation-snippet">{citation.snippet}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}
