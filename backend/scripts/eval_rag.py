"""RAG 评测：召回准确率 / 召回完整率 / 排序质量 / 生成答案质量。

和 tests/ 下的 Demo*.py 不同，这里不是单元测试，而是**端到端评测 harness**：
进程内直接调用 app.services.document_service.search_documents 与
app.services.qa_service.answer_question，绕开 HTTP 层，这样可以在一次运行里
切换 mode/rerank 多个配置，而不用重启服务。

评测集：backend/eval_work/testset.jsonl（人工编写 + grep 校验，见 build_testset.py）。
每道题的 gold 标注到「文档 + 逐字证据片段」两级，于是可以同时算文档级和片段级指标。

用法：
    .venv/Scripts/python.exe scripts/eval_rag.py                    # 全量
    .venv/Scripts/python.exe scripts/eval_rag.py --limit 5          # 冒烟
    .venv/Scripts/python.exe scripts/eval_rag.py --skip-gen         # 只跑检索
    .venv/Scripts/python.exe scripts/eval_rag.py --gen hybrid:true hybrid:false

指标口径（报告里会复述一遍，避免误读）：

  文档级相关（rel_doc）：命中的 chunk 属于 gold 文档之一。
  片段级相关（rel=2）：命中的 chunk（或它的父块）逐字包含该题的 gold 证据片段。
  片段级弱相关（rel=1）：属于 gold 文档但没含证据片段。
  其余 rel=0。

  - 召回准确率 P@k  = 相关命中数 / 返回命中数（分母用实际返回条数，重排开启时会被
    RERANK_TOP_K 压到 4，硬用 8 当分母会把截断算成不准确）
  - 召回完整率 R@k  = |命中的 gold 文档| / |该题 gold 文档总数|（多跳题才有分数值）
  - Hit@1 / Hit@3   = 前 1 / 前 3 条里有没有相关命中
  - MRR             = 第一条相关命中的名次倒数
  - nDCG@k          = 按 rel∈{0,1,2} 分级折损累计增益，理想序取「已返回集合」的
                      降序排列。**这是对返回列表内部排序质量的度量**，不是对全库
                      召回完备性的度量——后者受「只标注了 gold 文档」这个池化限制，
                      nDCG 在这里的绝对值会被高估，跨配置比较才有意义。
"""

from __future__ import annotations

import argparse
import json
import math
import logging
import statistics
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# 控制台是 GBK，中文指标名会乱码；强制 UTF-8，重定向到文件时也可读
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sqlalchemy import select  # noqa: E402

from app.Schemas import AskRequest, SearchHit  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app import services  # noqa: E402
from app.services.document_service import search_documents  # noqa: E402
from app.services.qa_service import answer_question  # noqa: E402
from app.services import model_factory  # noqa: E402
from app.services import qa_service  # noqa: E402
from app.services.model_factory import invoke_messages  # noqa: E402
from app.config import get_settings  # noqa: E402

# 上游（DeepSeek）实测约 1/5 的请求会卡在队列里，直到服务端 900s 上限才报错，
# 而正常请求只要 2~3s。不设客户端超时的话，一轮 ~270 次调用会被拖成几小时。
# 这里压到 45s，重试交给外层 _retry（带短退避），SDK 自己不再重试以免叠加等待。
# **只影响超时与重试，不改提示词、不改检索行为。**
REQUEST_TIMEOUT = 45.0


def build_model(settings, temperature=0.5):
    """与 model_factory.create_chat_model 同一套参数，外加超时上限。"""
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        model=settings.deepseek_model,
        model_provider="openai",
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


# qa_service 是 `from ... import create_chat_model` 绑定的名字，所以要改它模块上的属性
qa_service.create_chat_model = build_model

# database.py 里 echo=True 是无条件的，会在 import 时把 sqlalchemy.engine 打开。
# 必须放在 import 之后关，否则会被 create_engine 覆盖回去。
for _name in ("sqlalchemy.engine", "sqlalchemy.engine.Engine", "sqlalchemy.pool"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.WARNING)
    _lg.disabled = True

WORK = BACKEND_ROOT / "eval_work"
RESULTS = WORK / "results"

# 检索配置：(mode, rerank)。检索侧没有随机性，同一配置重跑结果一致
RETRIEVAL_CONFIGS = [
    ("hybrid", True),
    ("hybrid", False),
    ("vector", True),
    ("vector", False),
    ("bm25", True),
    ("bm25", False),
]

TOP_K = 8  # 与 /search、/qa/ask 的默认值一致


def load_testset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- 检索指标

def _hit_relevance(hit: SearchHit, item: dict) -> int:
    """0/1/2 三级相关性。片段级要认父块（small-to-big 下正文可能在父块里）。"""
    if hit.document_id not in item["gold_doc_ids"]:
        return 0
    text = f"{hit.content or ''}\n{hit.parent_content or ''}"
    for ev in item["evidence"]:
        if ev["span"] in text:
            return 2
    return 1


def _ndcg(gains: list[int], k: int) -> float:
    """分级 nDCG@k，理想序 = 已返回集合按增益降序。空列表返回 0。"""
    actual = gains[:k]
    ideal = sorted(gains, reverse=True)[:k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(actual))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(session, item: dict, mode: str, rerank: bool) -> dict:
    """跑一次检索，返回逐题的指标与中间结果。"""
    started = time.perf_counter()
    try:
        outcome = search_documents(session, item["question"], top_k=TOP_K, mode=mode, rerank=rerank)
        error = None
    except Exception as exc:  # 单题失败不该中断整轮评测
        return {"qid": item["qid"], "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000, "hits": []}
    latency_ms = (time.perf_counter() - started) * 1000

    hits = outcome.hits
    rels = [_hit_relevance(hit, item) for hit in hits]
    rel_docs = [r >= 1 for r in rels]
    gold_ids = set(item["gold_doc_ids"])

    retrieved_gold = {hit.document_id for hit in hits if hit.document_id in gold_ids}
    recall = len(retrieved_gold) / len(gold_ids) if gold_ids else None
    precision = (sum(rel_docs) / len(hits)) if hits else 0.0
    precision_span = (sum(1 for r in rels if r == 2) / len(hits)) if hits else 0.0

    first_rank = next((i + 1 for i, ok in enumerate(rel_docs) if ok), None)
    mrr = 1.0 / first_rank if first_rank else 0.0

    return {
        "qid": item["qid"],
        "status": outcome.status,
        "error": error,
        "latency_ms": latency_ms,
        "n_hits": len(hits),
        "n_relevant": sum(rel_docs),
        "rel_grades": rels,
        "recall": recall,
        "precision": precision,
        "precision_span": precision_span,
        "hit@1": bool(rel_docs[:1]) if rel_docs else False,
        "hit@3": any(rel_docs[:3]),
        "mrr": mrr,
        "ndcg": _ndcg(rels, TOP_K),
        "first_gold_rank": first_rank,
        "top_hit": {
            "document_id": hits[0].document_id if hits else None,
            "filename": (hits[0].section, hits[0].content[:60]) if hits else None,
            "rerank_score": hits[0].rerank_score if hits else None,
            "distance": hits[0].distance if hits else None,
            "bm25_score": hits[0].bm25_score if hits else None,
        },
        "hit_docs": [{"document_id": h.document_id, "rel": r,
                      "rerank_score": h.rerank_score, "distance": h.distance}
                     for h, r in zip(hits, rels)],
    }


ANSWERABLE = lambda items: [i for i in items if i["answerable"]]  # noqa: E731
UNANSWERABLE = lambda items: [i for i in items if not i["answerable"]]  # noqa: E731


def summarize_retrieval(rows: list[dict], items: list[dict]) -> dict:
    """把逐题结果聚合成一张配置级的表。不可答题只统计「拒答率」，不进召回指标。"""
    by_qid = {i["qid"]: i for i in items}
    ans_rows = [r for r in rows if by_qid[r["qid"]]["answerable"] and r["status"] != "error"]
    unans_rows = [r for r in rows if not by_qid[r["qid"]]["answerable"]]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    lat = [r["latency_ms"] for r in ans_rows]
    # 「完全召回」= 该题 gold 文档全部被召回（单跳题就是命中，多跳题要求两篇都中）
    full = [r for r in ans_rows if r["recall"] == 1.0]
    # 「完全无召回」= 一条相关都没返回（单跳题即为漏检）
    miss = [r for r in ans_rows if r["n_relevant"] == 0]
    return {
        "n_answerable": len(ans_rows),
        "recall": mean([r["recall"] for r in ans_rows]),
        "full_recall_rate": round(len(full) / len(ans_rows), 4) if ans_rows else None,
        "miss_rate": round(len(miss) / len(ans_rows), 4) if ans_rows else None,
        "precision": mean([r["precision"] for r in ans_rows]),
        "precision_span": mean([r["precision_span"] for r in ans_rows]),
        "hit@1": mean([1.0 if r["hit@1"] else 0.0 for r in ans_rows]),
        "hit@3": mean([1.0 if r["hit@3"] else 0.0 for r in ans_rows]),
        "mrr": mean([r["mrr"] for r in ans_rows]),
        "ndcg": mean([r["ndcg"] for r in ans_rows]),
        "avg_hits": mean([r["n_hits"] for r in ans_rows]),
        "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
        "latency_ms_p95": round(sorted(lat)[int(len(lat) * 0.95)], 1) if lat else None,
        "refusal_rate_unanswerable": mean([1.0 if r["status"] != "ok" else 0.0 for r in unans_rows]),
        "false_refusal_unanswerable": mean([1.0 if r["status"] == "ok" else 0.0 for r in unans_rows]),
        "answered_unanswerable_detail": {r["qid"]: r["status"] for r in unans_rows},
    }


# ---------------------------------------------------------------- 生成指标

JUDGE_PROMPT = """你是 RAG 系统的评测员。请对下面这次问答打分，只输出一个 JSON 对象，不要任何其他文字。

【问题】
{question}

【标准答案】(人工核实过，是唯一正确的口径)
{expected}

【检索到的证据】(模型只能看到这些材料)
{evidence}

【模型给出的回答】
{answer}

请按以下三项各打 1~5 分（整数）：
- correctness：回答与标准答案的事实一致性。数字/日期/单位/主体必须完全对得上；同义表述不影响得分。标准答案里没有的部分，模型答错了要扣分。
- faithfulness：回答里的每条事实是否都能在上述证据里找到依据。凭空补充外部知识、编造数字或文件名，都给 1~2 分。
- completeness：标准答案中的要点，回答覆盖了多少。

JSON 格式：{{"correctness": 4, "faithfulness": 5, "completeness": 4, "rationale": "一句话理由"}}"""


def parse_judge_json(text: str) -> dict | None:
    """judge 偶尔会带 ```json 围栏或前后废话，这里做一次宽松提取。"""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def judge_answer(model, item: dict, resp) -> dict:
    """LLM-as-judge。judge 看得到标准答案，所以只用于 correctness/completeness；
    faithfulness 的证据栏也一并给出，让它能在同一次调用里判定「有没有超纲」。"""
    evidence_block = ""
    for msg in resp.debug.messages:
        if msg.role == "human":
            evidence_block = msg.content
            break
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        expected=item["expected_answer"],
        evidence=evidence_block[:6000],
        answer=resp.answer,
    )
    try:
        content, _ = _retry(lambda: invoke_messages(model, [{"role": "human", "content": prompt}]))
        parsed = parse_judge_json(content)
        if parsed is None:
            return {"error": "judge 输出无法解析", "raw": content[:200]}
        return parsed
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _retry(fn, attempts: int = 3, backoff: float = 3.0):
    """上游排队/限流是常态，重试几次再放弃；最后一次的异常原样抛出。"""
    last = None
    for n in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if n < attempts - 1:
                time.sleep(backoff * (n + 1))
    raise last


def evaluate_generation(session, judge_model, item: dict, mode: str, rerank: bool) -> dict:
    started = time.perf_counter()
    payload = AskRequest(query=item["question"], top_k=TOP_K, mode=mode, rerank=rerank)
    try:
        resp = _retry(lambda: answer_question(session, payload))
    except Exception as exc:
        return {"qid": item["qid"], "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000}

    gold_ids = set(item["gold_doc_ids"])
    cited_docs = {c.document_id for c in resp.citations}
    row = {
        "qid": item["qid"],
        "status": resp.status,
        "question": item["question"],
        "answer": resp.answer,
        "answer_raw": resp.debug.answer_raw,
        "citations": [{"index": c.index, "document_id": c.document_id, "filename": c.filename}
                      for c in resp.citations],
        "citations_valid": resp.citations_valid,
        "invalid_citation_indices": resp.debug.invalid_citation_indices,
        "llm_called": resp.debug.llm_called,
        "llm_insufficient": resp.debug.llm_insufficient,
        "evidence_count": resp.debug.evidence_count,
        "evidence_scores": [e.rerank_score for e in resp.debug.evidence_scores],
        "usage": resp.debug.usage,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "retrieval_ms": resp.timings.retrieval_ms,
        "llm_ms": resp.timings.llm_ms,
        "total_ms": resp.timings.total_ms,
        "gold_doc_ids": item["gold_doc_ids"],
    }
    if item["answerable"]:
        row["citation_doc_precision"] = (
            len(cited_docs & gold_ids) / len(cited_docs) if cited_docs else None
        )
        row["citation_doc_recall"] = (
            len(cited_docs & gold_ids) / len(gold_ids) if gold_ids else None
        )
        row["answered"] = resp.status == "ok"
        if resp.status == "ok":
            row["judge"] = judge_answer(judge_model, item, resp)
        else:
            row["judge"] = None
    return row


def summarize_generation(rows: list[dict], items: list[dict]) -> dict:
    by_qid = {i["qid"]: i for i in items}
    ans = [r for r in rows if by_qid[r["qid"]]["answerable"] and r.get("status") != "error"]
    unans = [r for r in rows if not by_qid[r["qid"]]["answerable"] and r.get("status") != "error"]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    judged = [r["judge"] for r in ans if r.get("judge") and "correctness" in r["judge"]]
    answered = [r for r in ans if r.get("answered")]

    return {
        "n_answerable": len(ans),
        "answer_rate": round(len(answered) / len(ans), 4) if ans else None,
        "citations_valid_rate": mean([1.0 if r["citations_valid"] else 0.0 for r in answered]),
        "no_citation_rate": mean([1.0 if not r["citations"] else 0.0 for r in answered]),
        "citation_doc_precision": mean([r.get("citation_doc_precision") for r in answered]),
        "citation_doc_recall": mean([r.get("citation_doc_recall") for r in answered]),
        "judge_correctness": mean([j["correctness"] for j in judged]),
        "judge_faithfulness": mean([j["faithfulness"] for j in judged]),
        "judge_completeness": mean([j["completeness"] for j in judged]),
        "judge_correct_rate(>=4)": mean([1.0 if j["correctness"] >= 4 else 0.0 for j in judged]),
        "judged_n": len(judged),
        "llm_ms_mean": mean([r["llm_ms"] for r in ans]),
        "total_ms_mean": mean([r["total_ms"] for r in ans]),
        "abstention_rate_unanswerable": mean([1.0 if r["status"] != "ok" else 0.0 for r in unans]),
        "unanswerable_status": {r["qid"]: r["status"] for r in unans},
    }


# ---------------------------------------------------------------- 主流程

def warmup(session, items, use_llm: bool):
    """先跑一遍，把 embedding / reranker 的冷启动成本挡在评测之外。"""
    print("[warmup] 加载 embedding + reranker（首次较慢）…", flush=True)
    probe = items[0]["question"]
    t0 = time.perf_counter()
    search_documents(session, probe, top_k=TOP_K, mode="hybrid", rerank=True)
    print(f"[warmup] 检索就绪 {time.perf_counter() - t0:.1f}s", flush=True)
    if use_llm:
        t0 = time.perf_counter()
        model = build_model(get_settings(), temperature=0)
        invoke_messages(model, [{"role": "human", "content": "回复：ok"}])
        print(f"[warmup] 大模型就绪 {time.perf_counter() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(WORK / "testset.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--skip-gen", action="store_true", help="跳过生成质量评测（省 token）")
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="复用上一次同 tag 的检索结果，只重跑生成（改 judge 后省掉 12 分钟重排）")
    ap.add_argument("--gen", nargs="*", default=["hybrid:true", "hybrid:false"],
                    help="做生成评测的配置，形如 hybrid:true")
    ap.add_argument("--tag", default="run", help="结果文件名前缀")
    args = ap.parse_args()

    items = load_testset(Path(args.testset))
    if args.limit:
        items = items[:args.limit]
    print(f"测试集：{len(items)} 题（可答 {len(ANSWERABLE(items))}，不可答 {len(UNANSWERABLE(items))}）")

    RESULTS.mkdir(parents=True, exist_ok=True)
    session = SessionLocal()
    warmup(session, items, use_llm=not args.skip_gen)

    # ---- 阶段一：检索 ----
    retrieval_rows: dict[str, list[dict]] = {}
    retrieval_summary: dict[str, dict] = {}
    if args.skip_retrieval:
        prev_summary = json.loads((RESULTS / f"{args.tag}_summary.json").read_text(encoding="utf-8"))
        retrieval_summary = prev_summary["retrieval"]
        retrieval_rows = json.loads((RESULTS / f"{args.tag}_retrieval_rows.json").read_text(encoding="utf-8"))
        print("[retrieval] 复用上一次结果，跳过", flush=True)
    else:
        for mode, rerank in RETRIEVAL_CONFIGS:
            key = f"{mode}:{'rerank' if rerank else 'norerank'}"
            print(f"[retrieval] {key} …", flush=True)
            t0 = time.perf_counter()
            rows = [evaluate_retrieval(session, item, mode, rerank) for item in items]
            retrieval_rows[key] = rows
            retrieval_summary[key] = summarize_retrieval(rows, items)
            s = retrieval_summary[key]
            print(f"            R={s['recall']} P={s['precision']} MRR={s['mrr']} "
                  f"nDCG={s['ndcg']} 拒答率={s['refusal_rate_unanswerable']} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

        (RESULTS / f"{args.tag}_retrieval_rows.json").write_text(
            json.dumps(retrieval_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 阶段二：生成 ----
    generation_rows: dict[str, list[dict]] = {}
    generation_summary: dict[str, dict] = {}
    if not args.skip_gen:
        judge_model = build_model(get_settings(), temperature=0)
        for spec in args.gen:
            mode, rerank_s = spec.split(":")
            rerank = rerank_s.lower() in ("true", "1", "yes")
            key = f"{mode}:{'rerank' if rerank else 'norerank'}"
            print(f"[generation] {key} …", flush=True)
            t0 = time.perf_counter()
            rows = []
            for n, item in enumerate(items, 1):
                rows.append(evaluate_generation(session, judge_model, item, mode, rerank))
                if n % 10 == 0:
                    print(f"             {n}/{len(items)}  {time.perf_counter() - t0:.0f}s", flush=True)
            generation_rows[key] = rows
            generation_summary[key] = summarize_generation(rows, items)
            g = generation_summary[key]
            print(f"             答出率={g['answer_rate']} 引用有效={g['citations_valid_rate']} "
                  f"correct={g['judge_correctness']} faith={g['judge_faithfulness']} "
                  f"拒答率={g['abstention_rate_unanswerable']}", flush=True)

    summary = {
        "testset": str(args.testset),
        "n_items": len(items),
        "n_answerable": len(ANSWERABLE(items)),
        "n_unanswerable": len(UNANSWERABLE(items)),
        "top_k": TOP_K,
        "retrieval": retrieval_summary,
        "generation": generation_summary,
    }
    (RESULTS / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if generation_rows:
        (RESULTS / f"{args.tag}_generation_rows.json").write_text(
            json.dumps(generation_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
