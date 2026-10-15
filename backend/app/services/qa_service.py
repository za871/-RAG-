"""RAG 问答流水线：检索 → 拒答判定 → 证据组装 → 约束提示词 → 调大模型 → 引用校验。

这里是本项目唯一把「检索」和「大模型」接起来的地方，所以 DB / 网络 / 计时都集中在这一个文件，
纯文本契约（提示词、证据渲染、引用解析）全在 prompt.py 里。

**没有证据就不调大模型**——这是本服务最重要的一条：拒答路径直接返回，不构造 prompt、
不创建 model client。用检索侧已有的 status 判定，省一次调用、也彻底断了「让模型硬答」的可能。

流水线按「调大模型」这条线切成两半，好让一次问答和一次流式问答共用同一套判定：
`prepare_answer`（检索 → 拒答 → 证据 → 提示词 → 建 client）与
`finalize`（引用解析 → 校验 → 组装 AskResponse）。
`answer_question` 把两半用同步 `.invoke()` 串起来，`stream_answer` 用 `.stream()` 串起来，
所以「拒答条件」「引用校验」永远只有一份实现，两条路不可能给出不一样的 status。
"""

from dataclasses import dataclass, field
from datetime import date
from time import perf_counter
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.Schemas import (
    AskRequest,
    AskResponse,
    ChatMessage,
    Citation,
    DebugInfo,
    EvidenceScore,
    SearchHit,
    Timings,
)
from app.config import get_settings
from app.models import PolicyDocument
from app.services.document_service import (
    STATUS_OK,
    SearchOutcome,
    search_documents,
)
from app.services.model_factory import create_chat_model, invoke_messages, stream_messages
from app.services.prompt import (
    INSUFFICIENT_MARKER,
    Evidence,
    build_messages,
    parse_citations,
    render_evidence,
)

#引用 snippet 的截断长度
SNIPPET_CHARS = 120

REFUSAL_NO_EVIDENCE = "知识库中没有检索到与该问题相关的政策依据，因此无法作答。请补充问题细节或先上传相关政策文档。"
REFUSAL_REFUSED = "检索到的材料与你的问题相关性不足，为避免给出不准确的回答，本次不予作答。请换一种问法或补充背景信息。"
REFUSAL_INSUFFICIENT = "已检索到相关材料，但其中的信息不足以回答该问题，因此不予作答。请补充问题细节或提供更具体的问题。"
REASON_INSUFFICIENT_EVIDENCE = "证据不足以回答该问题"

#检索侧状态 → 罐装拒答文案
_REFUSAL_ANSWER = {
    "no_evidence": REFUSAL_NO_EVIDENCE,
    "refused": REFUSAL_REFUSED,
}


class LLMError(Exception):
    """大模型调用失败：网络不可达 / 鉴权失败 / 限流 / 模型名不存在。

    单独一个异常类型（继承 Exception 而不是 RuntimeError），让路由能把它映射成 502
    「上游不可用」，而把 RuntimeError（API key 没配、重排模型加载失败）映射成 503
    「我们自己的配置问题」——两者对调用方的含义不一样。
    """


def resolve_filenames(session: Session, document_ids: set[str]) -> dict[str, str]:
    """一次 select ... where id in (...) 拿回 document_id -> filename。

    引用要给人看，光有 document_id（uuid）没用，必须有文件名。
    SearchHit 上不带 filename，所以得回 MySQL 补这一下——和 _materialize_hits
    父块回查是同一个模式：先收 id 集合，再一次 in_ 查询。
    """
    if not document_ids:
        return {}
    rows = session.execute(
        select(PolicyDocument.id, PolicyDocument.filename)
        .where(PolicyDocument.id.in_(document_ids))
    ).all()
    return {row.id: row.filename for row in rows}


def _text_of(hit: SearchHit) -> str:
    """证据正文：优先父块（small-to-big 的上下文），没有父块就退回子块本身，绝不留空。"""
    return hit.parent_content or hit.content


def build_evidence(
        session: Session,
        hits: list[SearchHit],
        *,
        max_chars: int,
) -> tuple[list[Evidence], int]:
    """把命中折叠成证据，并做长度预算。返回 (证据列表, 被丢掉的条数)。

    折叠：同一父块的多个子块可能同时被召回（hybrid 下很常见），而 parent_content
    完全相同——不折叠就会把同一段父块重复喂给 LLM，还会产生重复引用。
    无父块的命中各自独立成块，key 用 child:<chunk_id>，不能把多个 None 当成同一个 key。
    """
    grouped: dict[str, list[SearchHit]] = {}
    order: list[str] = []
    for hit in hits:
        key = hit.parent_id or f"child:{hit.chunk_id}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(hit)

    #长度预算：超了就整条丢掉（不截断正文）。截断会破坏【证据 n】的完整性，
    #把证据切一半送进去比少给一条更糟。hits 已按相关度排好，所以丢的是尾巴上最不相关的。
    #第一条无条件保留：否则「检索说 ok 但证据为空」会出现，提示词里的「1 到 0」也没法看
    selected: list[list[SearchHit]] = []
    total_chars = 0
    dropped = 0
    for position, key in enumerate(order):
        text = _text_of(grouped[key][0])
        if selected and total_chars + len(text) > max_chars:
            dropped = len(order) - position
            break
        selected.append(grouped[key])
        total_chars += len(text)

    #filename 只对最终入选的证据查一次（折叠、预算都做完之后）
    filenames = resolve_filenames(
        session, {group[0].document_id for group in selected}
    )

    evidence: list[Evidence] = []
    for index, group in enumerate(selected, start=1):
        head = group[0]      # 该父块下排名最高的子块 = 代表块
        evidence.append(Evidence(
            index=index,
            chunk_id=head.chunk_id,
            hit_chunk_ids=[hit.chunk_id for hit in group],
            document_id=head.document_id,
            #查不到的降级成空串：理论不可达（并发删除），但不该让整条请求挂掉
            filename=filenames.get(head.document_id, ""),
            section=head.section,
            page=head.page,
            topic=head.topic,
            effective_date=head.effective_date,
            text=_text_of(head),
            rerank_score=head.rerank_score,
        ))
    return evidence, dropped


def _build_debug(
        *,
        settings,
        use_rerank: bool,
        outcome,
        hit_count: int,
        evidence: list[Evidence],
        dropped: int,
        llm_called: bool,
        messages: list[dict],
        usage: dict,
        cited: list[int],
        invalid: list[int],
        llm_insufficient: bool,
        answer_raw: str | None,
) -> DebugInfo:
    return DebugInfo(
        model=settings.deepseek_model,
        temperature=settings.qa_temperature,
        rerank=use_rerank,
        retrieval_status=outcome.status,
        retrieval_reason=outcome.reason,
        hit_count=hit_count,
        evidence_count=len(evidence),
        evidence_dropped=dropped,
        evidence_chars=sum(len(item.text) for item in evidence),
        evidence_scores=[
            EvidenceScore(index=item.index, chunk_id=item.chunk_id,
                          rerank_score=item.rerank_score)
            for item in evidence
        ],
        llm_called=llm_called,
        #逐字回放真正发出去的那两条消息
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        usage=usage,
        cited_indices=cited,
        invalid_citation_indices=invalid,
        llm_insufficient=llm_insufficient,
        answer_raw=answer_raw,
    )


#SSE 事件名。常量放在服务层，是为了让「产出事件」和「渲染事件」的调用方共用同一组名字；
#真正的帧格式（event:/data: 怎么拼）留在 api/qa.py——服务层不该知道 HTTP 长什么样
EVENT_META = "meta"
EVENT_DELTA = "delta"
EVENT_DONE = "done"
EVENT_ERROR = "error"


@dataclass
class PreparedAnswer:
    """`prepare_answer` 的产物：要么是一次拒答，要么是一份「可以调大模型」的状态。

    刻意只装纯数据（SearchHit / Evidence / dict / 已建好的 model client），
    **不装任何 SQLAlchemy 对象**：流式路由会在 prepare_answer 返回之后、在另一个线程里
    才消费它（见 api/qa.py 里关于 Session 跨线程的注释）。
    """

    #非 None 即拒答：此时下面所有字段都没意义，调用方应当直接把它返回给前端
    refusal: AskResponse | None = None

    query: str = ""
    settings: Any = None
    use_rerank: bool = False
    outcome: SearchOutcome | None = None
    evidence: list[Evidence] = field(default_factory=list)
    #因上下文预算被整条丢掉的证据数
    dropped: int = 0
    hit_count: int = 0
    messages: list[dict] = field(default_factory=list)
    model: Any = None
    retrieval_ms: float = 0.0
    #流水线起点，total_ms 从它算起（含证据组装、filename 回查、引用解析）
    started: float = 0.0


def prepare_answer(session: Session, payload: AskRequest) -> PreparedAnswer:
    """跑「调大模型之前」的全部步骤，并把所有 DB 访问都收在这个函数里。

    拒答时**不建 model client**——本服务最重要的那条约束在这里落地一次，
    /ask 和 /ask/stream 就都自动继承，不存在「流式那条路忘了判」的可能。
    """
    settings = get_settings()
    #先把 None 解析成具体 bool 再往下传：debug 里的值就一定和实际行为一致，
    #不会出现「debug 说开着、实际没开」这种最没用的调试信息
    use_rerank = settings.enable_reranker if payload.rerank is None else payload.rerank

    started = perf_counter()
    retrieval_started = perf_counter()
    outcome = search_documents(
        session,
        payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
        topic=payload.topic,
        effective_before=payload.effective_before,
        mode=payload.mode,
        rerank=use_rerank,
    )
    retrieval_ms = (perf_counter() - retrieval_started) * 1000

    #没证据就拒答：直接返回，不构造 prompt、不建 model client
    if outcome.status != STATUS_OK:
        return PreparedAnswer(refusal=AskResponse(
            query=payload.query,
            status=outcome.status,
            reason=outcome.reason,
            answer=_REFUSAL_ANSWER.get(outcome.status, REFUSAL_NO_EVIDENCE),
            citations=[],
            citations_valid=True,
            timings=Timings(
                retrieval_ms=round(retrieval_ms, 2),
                llm_ms=0.0,
                total_ms=round((perf_counter() - started) * 1000, 2),
            ),
            debug=_build_debug(
                settings=settings, use_rerank=use_rerank, outcome=outcome,
                hit_count=0, evidence=[], dropped=0,
                llm_called=False, messages=[], usage={},
                cited=[], invalid=[], llm_insufficient=False, answer_raw=None,
            ),
        ))

    evidence, dropped = build_evidence(
        session, outcome.hits, max_chars=settings.qa_max_context_chars
    )
    messages = build_messages(
        len(evidence), payload.query, render_evidence(evidence)
    )

    #stream_usage=True 对非流式路径没有任何影响：stream_options 只在 _stream/_astream 里
    #才被拼进请求，.invoke() 的 payload 与不传它时逐字节一致（已实测两者的 keys 相同）。
    #显式传它是为了让流式路径也拿得到 usage_metadata——不传就恒为空，见 model_factory
    model = create_chat_model(
        settings, temperature=settings.qa_temperature, stream_usage=True
    )

    return PreparedAnswer(
        query=payload.query,
        settings=settings,
        use_rerank=use_rerank,
        outcome=outcome,
        evidence=evidence,
        dropped=dropped,
        hit_count=len(outcome.hits),
        messages=messages,
        model=model,
        retrieval_ms=retrieval_ms,
        started=started,
    )


def finalize(
        content: str,
        usage: dict,
        prepared: PreparedAnswer,
        llm_ms: float,
) -> AskResponse:
    """把「模型输出」+「准备阶段的状态」组装成最终响应。

    引用解析、不足判定、引用校验、debug 组装全在这里，**只有这一份实现**——
    /ask 和 /ask/stream 都从这里出去，两条路的 status 语义和 citations 不可能不一致。
    """
    parsed = parse_citations(content, len(prepared.evidence))
    #用子串而不是全等：模型经常在标记前后带一句中文客套话
    insufficient = INSUFFICIENT_MARKER in content

    if insufficient:
        status: str = "insufficient_evidence"
        reason: str | None = REASON_INSUFFICIENT_EVIDENCE
        answer = REFUSAL_INSUFFICIENT
        citations: list[Citation] = []
        citations_valid = True
    else:
        status = "ok"
        reason = None
        answer = parsed.text
        citations = [
            Citation(
                index=item.index,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                section=item.section,
                page=item.page,
                topic=item.topic,
                effective_date=item.effective_date,
                snippet=item.text[:SNIPPET_CHARS] + ("…" if len(item.text) > SNIPPET_CHARS else ""),
                rerank_score=item.rerank_score,
            )
            for item in (prepared.evidence[number - 1] for number in parsed.cited)
        ]
        #零引用也算不合法：提示词要求「每条事实陈述必须标注引用」，
        #而一个非拒答的答案却没有任何引用 = 整段无据，这正是要暴露的异常
        citations_valid = not parsed.invalid and len(parsed.cited) > 0

    return AskResponse(
        query=prepared.query,
        status=status,
        reason=reason,
        answer=answer,
        citations=citations,
        citations_valid=citations_valid,
        timings=Timings(
            retrieval_ms=round(prepared.retrieval_ms, 2),
            llm_ms=round(llm_ms, 2),
            #total 含证据组装、filename 回查、引用解析，所以一定 >= 前两者之和
            total_ms=round((perf_counter() - prepared.started) * 1000, 2),
        ),
        debug=_build_debug(
            settings=prepared.settings, use_rerank=prepared.use_rerank,
            outcome=prepared.outcome, hit_count=prepared.hit_count,
            evidence=prepared.evidence, dropped=prepared.dropped,
            llm_called=True, messages=prepared.messages, usage=usage,
            cited=parsed.cited, invalid=parsed.invalid,
            llm_insufficient=insufficient,
            #原始输出只在真调了模型时才有意义
            answer_raw=content,
        ),
    )


def answer_question(session: Session, payload: AskRequest) -> AskResponse:
    """跑完整条 RAG 流水线（一次调用、一次性返回）。异常语义见 LLMError 的注释。"""
    prepared = prepare_answer(session, payload)
    if prepared.refusal is not None:
        return prepared.refusal

    llm_started = perf_counter()
    try:
        content, usage = invoke_messages(prepared.model, prepared.messages)
    except Exception as exc:
        #RuntimeError（没配 key）不在这里，它在 create_chat_model 里就抛了，语义不同
        raise LLMError(
            f"调用大模型失败（model={prepared.settings.deepseek_model}）：{exc}"
        ) from exc
    llm_ms = (perf_counter() - llm_started) * 1000

    return finalize(content, usage, prepared, llm_ms)


def stream_answer(prepared: PreparedAnswer) -> Iterator[tuple[str, dict]]:
    """把「已准备好的」问答跑成事件流：(事件名, 可直接 JSON 序列化的 payload)。

    只接收 PreparedAnswer、**不接收 Session**：生成器的每次 next() 都跑在
    StreamingResponse 的线程池线程里，而 Session 是在处理请求的那个线程里建的。
    把 DB 访问全留在 prepare_answer，这条「Session 跨线程」的隐患就从设计上不存在
    （对照 api/qa.py 里对 async def + run_in_threadpool 的那段权衡）。
    """
    if prepared.refusal is not None:
        #拒答不发 meta/delta：没有正在生成的正文，前端拿 done 里的完整响应渲染即可。
        #这也让前端两条路径共用一套渲染——拒答的响应体和 /ask 是同一个模型序列化出来的
        yield (EVENT_DONE, prepared.refusal.model_dump(mode="json"))
        return

    #先把检索结果抖出去：重排冷启动要加载 ~2.2G 权重（首次 6 秒+），
    #等第一个 token 的这段时间里，界面至少能显示「已检索到 N 条证据，正在生成」
    yield (EVENT_META, {
        "status": STATUS_OK,
        "retrieval_ms": round(prepared.retrieval_ms, 2),
        "hit_count": prepared.hit_count,
        "evidence_count": len(prepared.evidence),
        "evidence_dropped": prepared.dropped,
        "rerank": prepared.use_rerank,
        "model": prepared.settings.deepseek_model,
    })

    usage: dict = {}
    llm_started = perf_counter()
    parts: list[str] = []
    try:
        for delta in stream_messages(prepared.model, prepared.messages, usage):
            parts.append(delta)
            yield (EVENT_DELTA, {"text": delta})
    except Exception as exc:
        #响应头早就发出去了（状态码已经是 200），改不了状态码，只能发一个 error 事件。
        #文案和 /ask 的 502 保持同一句，方便两边对照排查
        yield (EVENT_ERROR, {
            "detail": f"调用大模型失败（model={prepared.settings.deepseek_model}）：{exc}"
        })
        return

    llm_ms = (perf_counter() - llm_started) * 1000
    #用拼起来的增量走和 /ask 完全相同的 finalize：引用校验、拒答判定都不重写一遍
    response = finalize("".join(parts), usage, prepared, llm_ms)
    yield (EVENT_DONE, response.model_dump(mode="json"))
