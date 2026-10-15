import { useCallback, useState } from 'react'

import AnswerView from './components/AnswerView'
import CitationList from './components/CitationList'
import DebugDetails from './components/DebugDetails'
import QuestionForm from './components/QuestionForm'
import StatusNotice from './components/StatusNotice'
import TimingBar from './components/TimingBar'
import { useAskStream } from './hooks/useAskStream'

export default function App() {
  const { state, ask, stop, reset, busy } = useAskStream()
  const [highlight, setHighlight] = useState<number | null>(null)

  const handleCite = useCallback((n: number) => {
    setHighlight(n)
    document
      .getElementById(`citation-${n}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [])

  const { phase, result, text, meta, error, errorFromServer, query } = state
  const showResult = phase !== 'idle'
  // 拒答的文案已经由 StatusNotice 呈现（连罐装答复带 reason 一起），正文区不再重复一遍
  const showAnswerBody = phase === 'done' && result?.status === 'ok'
  const hasPartialText = phase === 'stopped' || phase === 'error'

  return (
    <div className="app">
      <header className="app-header">
        <h1>政务政策知识库问答</h1>
        <p className="app-subtitle">
          答案只依据知识库中的政策原文生成，每条事实都标注引用编号
        </p>
      </header>

      <QuestionForm
        busy={busy}
        hasResult={showResult}
        onSubmit={ask}
        onStop={stop}
        onReset={reset}
      />

      {phase === 'retrieving' && (
        <div className="skeleton">
          <div className="skeleton-bar" />
          <p>
            正在检索知识库…
            <span className="skeleton-hint">
              （服务刚启动或首次提问时，重排模型要加载约 2G 权重，可能等待 10 秒以上）
            </span>
          </p>
        </div>
      )}

      {showResult && (
        <section className="result">
          <div className="result-question">{query}</div>

          {error && (
            <div className="notice notice-error">
              <span className="notice-icon">✕</span>
              <div>
                <strong>{errorFromServer ? '请求被拒绝' : '请求失败'}</strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          {/* 拒答与校验异常都走这里；status=ok 且引用合法时它自己不渲染 */}
          {result && <StatusNotice result={result} />}

          {(phase === 'streaming' || phase === 'retrieving') && (
            <AnswerView
              text={text}
              citations={null}
              streaming={phase === 'streaming'}
              highlight={highlight}
              onCite={handleCite}
            />
          )}

          {showAnswerBody && (
            <AnswerView
              text={text}
              citations={result.citations}
              streaming={false}
              highlight={highlight}
              onCite={handleCite}
            />
          )}

          {/* 中途停下/出错时，已经收到的正文仍然有用，保留下来自己看 */}
          {hasPartialText && text && (
            <>
              <div className="partial-note">
                {phase === 'stopped' ? '已停止生成，以下是已收到的部分内容' : '以下是中断前已收到的内容'}
              </div>
              <AnswerView
                text={text}
                citations={null}
                streaming={false}
                highlight={highlight}
                onCite={handleCite}
              />
            </>
          )}

          {result && (
            <CitationList
              citations={result.citations}
              highlight={highlight}
              onHover={setHighlight}
            />
          )}

          {(result || meta) && <TimingBar result={result} meta={meta} />}
          {result && <DebugDetails result={result} />}
        </section>
      )}
    </div>
  )
}
