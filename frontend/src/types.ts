/**
 * 后端 app/Schemas.py 的手抄版。
 *
 * 抄而不是自动生成（openapi-typescript 之类）：这套接口只有 5 个模型、字段全都见过，
 * 而手抄能让「后端改了字段前端没跟上」在 tsc 阶段就炸出来，比生成器的 diff 更好读。
 * 唯一的纪律是：Schemas.py 改了这里必须跟着改。
 */

/** 检索模式。与 SearchRequest.mode 的 Literal 对齐 */
export type RetrievalMode = 'hybrid' | 'vector' | 'bm25'

/** 非 ok 就是拒答。三种原因语义不同，前端要分开呈现 */
export type AnswerStatus = 'ok' | 'no_evidence' | 'refused' | 'insufficient_evidence'

/** 问答入参，字段与 AskRequest 一一对应 */
export interface AskRequest {
  query: string
  top_k?: number
  document_id?: string | null
  topic?: string | null
  effective_before?: string | null
  mode?: RetrievalMode
  /**
   * 三态，和 Schemas.py 里 `bool | None` 对齐：
   * 省略表示「跟随服务端 ENABLE_RERANKER 配置」，而不是 false。
   */
  rerank?: boolean | null
}

/** 一条被答案真正引用到的证据。index 就是正文里角标 [n] 的 n */
export interface Citation {
  index: number
  chunk_id: string
  document_id: string
  filename: string
  section: string | null
  page: number | null
  topic: string | null
  effective_date: string | null
  /** 证据正文前 120 字 */
  snippet: string
  rerank_score: number | null
}

export interface Timings {
  /** 检索总耗时；重排算在里面 */
  retrieval_ms: number
  /** 大模型耗时；拒答路径恒为 0 */
  llm_ms: number
  total_ms: number
}

export interface ChatMessage {
  role: string
  content: string
}

export interface EvidenceScore {
  index: number
  chunk_id: string
  rerank_score: number | null
}

export interface DebugInfo {
  model: string
  temperature: number
  /** 实际生效的重排开关（None 已在服务端解析成具体 bool） */
  rerank: boolean
  retrieval_status: string
  retrieval_reason: string | null
  /** 去重前的命中数 */
  hit_count: number
  evidence_count: number
  /** 因上下文预算被整条丢掉的证据数 */
  evidence_dropped: number
  evidence_chars: number
  evidence_scores: EvidenceScore[]
  llm_called: boolean
  /** system + user 的完整原文，逐字回放 */
  messages: ChatMessage[]
  usage: Record<string, unknown>
  cited_indices: number[]
  invalid_citation_indices: number[]
  llm_insufficient: boolean
  /** 模型原始输出（角标剔除前）；只有真调了模型才有值 */
  answer_raw: string | null
}

export interface AskResponse {
  query: string
  status: AnswerStatus
  reason: string | null
  /** status != ok 时这里是服务端的罐装拒答文案，不是模型输出的 */
  answer: string
  /** status != ok 时恒为空（Schemas.py 的 model_validator 保证） */
  citations: Citation[]
  /** 只对 status=ok 有意义：要求「至少一条引用」且「没有越界引用」 */
  citations_valid: boolean
  timings: Timings
  debug: DebugInfo
}

/** SSE 的 meta 事件：检索完成、第一个 token 还没来时的进度信息 */
export interface StreamMeta {
  status: string
  retrieval_ms: number
  hit_count: number
  evidence_count: number
  evidence_dropped: number
  rerank: boolean
  model: string
}

export type StreamEvent =
  | { event: 'meta'; data: StreamMeta }
  | { event: 'delta'; data: { text: string } }
  | { event: 'done'; data: AskResponse }
  | { event: 'error'; data: { detail: string } }
