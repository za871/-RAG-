import { Fragment } from 'react'

import type { Citation } from '../types'

interface Props {
  text: string
  /** done 之后才有值。流式期间是 null，此时角标不可点（还不存在 n → 证据 的映射） */
  citations: Citation[] | null
  streaming: boolean
  highlight: number | null
  onCite: (n: number) => void
}

/** 行内角标：正文里的 [n] 渲染成可点的小方块，点了就滚到对应的引用卡片 */
function renderInline(
  line: string,
  known: Map<number, Citation> | null,
  highlight: number | null,
  onCite: (n: number) => void,
) {
  // 用捕获组切分，分隔符本身会留在结果数组里——这样不需要手工做位置推算
  const parts = line.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const matched = /^\[(\d+)\]$/.exec(part)
    if (!matched) {
      return <Fragment key={i}>{part}</Fragment>
    }
    const n = Number(matched[1])
    const clickable = known?.has(n) ?? false
    return (
      <button
        key={i}
        type="button"
        className={`cite-mark${highlight === n ? ' is-active' : ''}${clickable ? '' : ' is-plain'}`}
        // 流式期间还没有 citations，点了也没有目标可跳；禁用比点了没反应更好懂
        disabled={!clickable}
        title={clickable ? `跳转到引用 [${n}]` : `引用 [${n}]`}
        onClick={() => onCite(n)}
      >
        {n}
      </button>
    )
  })
}

/**
 * 答案正文。
 *
 * 后端 parse_citations 会**保留**合法的 [n]、只剔除越界的（prompt.py:120-125），
 * 所以正文里的角标和 citations 天然对得上，前端只做渲染、不做二次判定。
 */
export default function AnswerView({ text, citations, streaming, highlight, onCite }: Props) {
  const known = citations ? new Map(citations.map((c) => [c.index, c])) : null
  // 空行分段；段内单个换行当软换行处理。政策文本经常是「先结论再分点」，
  // 不按空行分段会挤成一大坨，很影响读
  const paragraphs = text.split(/\n{2,}/).filter((p) => p.trim())

  return (
    <div className="answer">
      {paragraphs.map((paragraph, pi) => (
        <p key={pi}>
          {paragraph.split('\n').map((line, li) => (
            <Fragment key={li}>
              {li > 0 && <br />}
              {renderInline(line, known, highlight, onCite)}
            </Fragment>
          ))}
        </p>
      ))}
      {streaming && <span className="caret" aria-hidden="true" />}
    </div>
  )
}
