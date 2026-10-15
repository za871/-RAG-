import { useState, type FormEvent, type KeyboardEvent } from 'react'

import type { AskRequest, RetrievalMode } from '../types'

interface Props {
  busy: boolean
  hasResult: boolean
  onSubmit: (payload: AskRequest) => void
  onStop: () => void
  onReset: () => void
}

const EXAMPLES = [
  '人才引进有哪些补贴政策？',
  '申请需要提交哪些材料？',
  '政策的有效期限是多久？',
]

/** rerank 是三态（Schemas.py 里是 bool | None），用一个 select 表达，'default' 表示不传 */
type RerankChoice = 'default' | 'on' | 'off'

export default function QuestionForm({ busy, hasResult, onSubmit, onStop, onReset }: Props) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<RetrievalMode>('hybrid')
  const [rerank, setRerank] = useState<RerankChoice>('default')
  const [topK, setTopK] = useState('8')
  const [topic, setTopic] = useState('')
  const [effectiveBefore, setEffectiveBefore] = useState('')
  const [documentId, setDocumentId] = useState('')

  const submit = (event?: FormEvent) => {
    event?.preventDefault()
    const text = query.trim()
    if (!text || busy) return

    const payload: AskRequest = { query: text }
    // 只带上真正填了的字段。后端的模型是 extra="forbid"，而且
    // effective_before: "" 这种空串会被 FastAPI 当成非法日期直接 422——
    // 与其让用户看到一个莫名其妙的校验错误，不如什么都不传（不传就是「不按此过滤」）
    if (mode !== 'hybrid') payload.mode = mode
    if (rerank !== 'default') payload.rerank = rerank === 'on'
    const parsedTopK = Number(topK)
    if (Number.isFinite(parsedTopK) && parsedTopK !== 8) payload.top_k = parsedTopK
    if (topic.trim()) payload.topic = topic.trim()
    if (effectiveBefore) payload.effective_before = effectiveBefore
    if (documentId.trim()) payload.document_id = documentId.trim()

    onSubmit(payload)
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl/Cmd + Enter 提交，单 Enter 留给换行——问题是可能贴着多行原文粘贴进来的
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') submit()
  }

  return (
    <form className="ask-form" onSubmit={submit}>
      <textarea
        className="ask-input"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="就政策内容提问，例如：人才引进有哪些补贴政策？"
        rows={3}
        disabled={busy}
      />

      <div className="ask-actions">
        {busy ? (
          <button type="button" className="btn btn-ghost" onClick={onStop}>
            停止生成
          </button>
        ) : (
          <button type="submit" className="btn btn-primary" disabled={!query.trim()}>
            提问
          </button>
        )}
        {hasResult && !busy && (
          <button type="button" className="btn btn-ghost" onClick={onReset}>
            清空
          </button>
        )}
        <span className="ask-hint">Ctrl / ⌘ + Enter 提交</span>
      </div>

      {!hasResult && (
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="example-chip"
              disabled={busy}
              onClick={() => setQuery(example)}
            >
              {example}
            </button>
          ))}
        </div>
      )}

      <details className="advanced">
        <summary>高级选项</summary>
        <div className="advanced-grid">
          <label>
            检索模式
            <select value={mode} onChange={(e) => setMode(e.target.value as RetrievalMode)}>
              <option value="hybrid">hybrid（向量 + BM25 融合）</option>
              <option value="vector">vector（纯语义）</option>
              <option value="bm25">bm25（纯关键词）</option>
            </select>
          </label>

          <label>
            重排
            <select value={rerank} onChange={(e) => setRerank(e.target.value as RerankChoice)}>
              <option value="default">跟随服务端配置</option>
              <option value="on">强制开启</option>
              {/* 关掉重排就没有相关性门槛了（MIN_EVIDENCE_SCORE 只作用于重排分），
                  无关问题会变成「ok + 一堆垃圾证据」，这里明确写出来 */}
              <option value="off">强制关闭（无相关性门槛）</option>
            </select>
          </label>

          <label>
            top_k
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
            />
          </label>

          <label>
            主题（精确匹配）
            <input
              type="text"
              maxLength={100}
              value={topic}
              placeholder="如：人才引进"
              onChange={(e) => setTopic(e.target.value)}
            />
          </label>

          <label>
            只取生效日期不晚于
            <input
              type="date"
              value={effectiveBefore}
              onChange={(e) => setEffectiveBefore(e.target.value)}
            />
          </label>

          <label>
            限定文档 ID
            <input
              type="text"
              value={documentId}
              placeholder="留空则在全库检索"
              onChange={(e) => setDocumentId(e.target.value)}
            />
          </label>
        </div>
      </details>
    </form>
  )
}
