"""把 eval_rag.py 的结果 JSON 渲染成 Markdown 报告。

只做「表 + 可复现的事实」，结论性的解读留给人工写，免得结论和数字脱节。
用法：
    .venv/Scripts/python.exe scripts/eval_report.py --tag main --out eval_work/RAG评测报告.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
WORK = BACKEND_ROOT / "eval_work"
RESULTS = WORK / "results"

CONFIG_LABEL = {
    "hybrid:rerank": "hybrid + 重排（生产默认）",
    "hybrid:norerank": "hybrid（无重排）",
    "vector:rerank": "vector + 重排",
    "vector:norerank": "vector（无重排）",
    "vector:norerank": "vector（无重排）",
    "bm25:rerank": "bm25 + 重排",
    "bm25:norerank": "bm25（无重排）",
}
QTYPE_LABEL = {
    "factoid": "事实型（措辞贴近原文）",
    "paraphrase": "同义改写（换掉关键词）",
    "numeric": "数字/日期型",
    "multi_hop": "多跳（需跨文档）",
    "list": "列举型",
    "unanswerable": "不可答（应拒答）",
}


def fmt(v, digits=3, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.{digits}f}"


def load(tag):
    summary = json.loads((RESULTS / f"{tag}_summary.json").read_text(encoding="utf-8"))
    ret_rows = json.loads((RESULTS / f"{tag}_retrieval_rows.json").read_text(encoding="utf-8"))
    gen_rows = {}
    p = RESULTS / f"{tag}_generation_rows.json"
    if p.exists():
        gen_rows = json.loads(p.read_text(encoding="utf-8"))
    items = [json.loads(l) for l in (WORK / "testset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return summary, ret_rows, gen_rows, items


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def retrieval_table(summary):
    keys = ["hybrid:rerank", "hybrid:norerank", "vector:rerank", "vector:norerank",
            "bm25:rerank", "bm25:norerank"]
    headers = ["配置", "召回完整率 R@8", "完全召回率", "完全漏检率", "召回准确率 P@8",
               "片段级准确率", "Hit@1", "Hit@3", "MRR", "nDCG@8", "平均命中数", "平均耗时"]
    rows = []
    for k in keys:
        s = summary["retrieval"].get(k)
        if not s:
            continue
        rows.append([
            CONFIG_LABEL.get(k, k),
            fmt(s["recall"], pct=True), fmt(s["full_recall_rate"], pct=True),
            fmt(s["miss_rate"], pct=True), fmt(s["precision"], pct=True),
            fmt(s["precision_span"], pct=True), fmt(s["hit@1"], pct=True),
            fmt(s["hit@3"], pct=True), fmt(s["mrr"]), fmt(s["ndcg"]),
            fmt(s["avg_hits"], 1), f"{s['latency_ms_mean']:.0f} ms",
        ])
    return table(headers, rows)


def per_type_table(summary):
    """按题型拆开看同一配置的召回表现，找「哪种问题最难」。"""
    keys = ["hybrid:rerank", "hybrid:norerank", "vector:norerank", "bm25:norerank"]
    headers = ["题型", "题数"] + [f"{CONFIG_LABEL[k]}<br>R@8 / MRR / nDCG" for k in keys]
    return headers, keys


def build(tag: str) -> str:
    summary, ret_rows, gen_rows, items = load(tag)
    by_qid = {i["qid"]: i for i in items}
    L = []
    A = L.append

    A("# 政务政策 RAG 系统 · 检索与生成质量评测报告")
    A("")
    A(f"- 评测集：`backend/eval_work/testset.jsonl`（{summary['n_items']} 题：可答 "
      f"{summary['n_answerable']} 题 + 不可答 {summary['n_unanswerable']} 题）")
    A(f"- 语料：MySQL/Milvus 中已入库的 **105 篇**河南工学院政策文档"
      f"（`hait_docs` 共 429 篇，本次仅入库 105 篇）")
    A(f"- 检索入口：`app.services.document_service.search_documents`，top_k = {summary['top_k']}"
      f"（= `/search`、`/qa/ask` 默认值）")
    A(f"- 生成入口：`app.services.qa_service.answer_question`")
    A("- 重排模型：BAAI/bge-reranker-v2-m3（CPU）；向量模型：bge-small-zh-v1.5（CPU）")
    A("- 生成模型：deepseek-v4-flash，temperature=0.2（问答）/ 0（评测员）")
    A("- 评测日期：2026-10-15")
    A("")

    # ---------------- 测试集构成 ----------------
    A("## 1. 测试集构成")
    A("")
    qc = Counter(i["qtype"] for i in items)
    A(table(["题型", "题数", "设计意图"],
            [[QTYPE_LABEL.get(k, k), v, INTENT.get(k, "")] for k, v in qc.most_common()]))
    A("")
    dc = Counter(i["difficulty"] for i in items)
    A(f"难度分布：易 {dc.get('easy', 0)} / 中 {dc.get('medium', 0)} / 难 {dc.get('hard', 0)}。")
    A("")
    multi = [i for i in items if len(i["gold_doc_indices"]) > 1]
    A(f"其中 **{len(multi)} 题是多 gold 文档**（多跳题，或同一个问题在多篇文档里都有对应条款，"
      f"如省级科技项目的年龄上限样板句同时出现在两篇通知里）。")
    A("")
    A("**每道题的 gold 标注经过逐条校验**：`evidence_spans` 必须是 gold 文档正文里"
      "**逐字出现**的连续片段，由 `build_testset.py` 用字符串包含关系复核，"
      "跨文档重复的样板句会把重复文档一并纳入 gold。")
    A("")

    # ---------------- 检索 ----------------
    A("## 2. 检索质量")
    A("")
    A(f"### 2.1 六种配置总览")
    A("")
    A(retrieval_table(summary))
    A("")
    A("指标口径：")
    A("")
    A("- **召回完整率 R@8**：命中的 gold 文档数 / 该题 gold 文档总数。多跳题才有分数值，"
      "单跳题非 0 即 1。")
    A("- **完全召回率**：该题 gold 文档**全部**被召回的题目占比。")
    A("- **完全漏检率**：一条相关命中都没有的题目占比（真正的「检索失败」）。")
    A("- **召回准确率 P@8**：返回结果里属于 gold 文档的条数占比。分母用**实际返回条数**——"
      "重排开启时 `RERANK_TOP_K=4` 会把命中压到最多 4 条，硬拿 8 当分母会把截断误算成不准确。")
    A("- **片段级准确率**：返回结果里**逐字包含 gold 证据片段**的条数占比，比文档级更严。")
    A("- **nDCG@8**：按 0/1/2 三级相关性算的分级折损累计增益，理想序取已返回集合的降序排列。"
      "**它衡量的是返回列表内部的排序质量**；因为只标注了 gold 文档（池化问题），"
      "绝对值偏高，只适合跨配置比较。")
    A("")

    # 分题型
    A("### 2.2 分题型（看哪种问题最难）")
    A("")
    keys = ["hybrid:rerank", "hybrid:norerank", "vector:norerank", "bm25:norerank"]
    per_type = defaultdict(lambda: defaultdict(list))
    for k in keys:
        for r in ret_rows.get(k, []):
            it = by_qid[r["qid"]]
            if not it["answerable"] or r["status"] == "error":
                continue
            per_type[it["qtype"]][k].append(r)
    headers = ["题型", "题数"] + [f"{CONFIG_LABEL[k]}<br>R@8 / MRR / nDCG" for k in keys]
    rows = []
    for qt in ["factoid", "paraphrase", "numeric", "multi_hop", "list"]:
        n = len(per_type[qt][keys[0]])
        if not n:
            continue
        cells = []
        for k in keys:
            rs = per_type[qt][k]
            cells.append(f"{fmt(statistics.fmean([r['recall'] for r in rs]), pct=True)} / "
                         f"{fmt(statistics.fmean([r['mrr'] for r in rs]))} / "
                         f"{fmt(statistics.fmean([r['ndcg'] for r in rs]))}")
        rows.append([QTYPE_LABEL[qt], n] + cells)
    A(table(headers, rows))
    A("")

    # ---------------- 生成 ----------------
    if summary.get("generation"):
        A("## 3. 生成答案质量")
        A("")
        A("生成侧跑的是完整链路：检索 → 拒答判定 → 证据组装 → 提示词 → 大模型 → 引用校验。")
        A("")
        gkeys = [k for k in ["hybrid:rerank", "hybrid:norerank"] if k in summary["generation"]]
        headers = ["指标"] + [CONFIG_LABEL[k] for k in gkeys]
        metrics = [
            ("答出率（status=ok 的占比）", "answer_rate", True),
            ("引用有效率 citations_valid", "citations_valid_rate", True),
            ("零引用率", "no_citation_rate", True),
            ("引用指向 gold 文档的准确率", "citation_doc_precision", True),
            ("引用覆盖 gold 文档的比例", "citation_doc_recall", True),
            ("评测员：正确性（1-5）", "judge_correctness", False),
            ("评测员：忠实度（1-5）", "judge_faithfulness", False),
            ("评测员：完整性（1-5）", "judge_completeness", False),
            ("正确率（正确性≥4 的占比）", "judge_correct_rate(>=4)", True),
            ("不可答题拒答率", "abstention_rate_unanswerable", True),
            ("平均 LLM 耗时", "llm_ms_mean", False),
            ("平均端到端耗时", "total_ms_mean", False),
        ]
        rows = []
        for label, key, pct in metrics:
            cells = []
            for k in gkeys:
                v = summary["generation"][k].get(key)
                cells.append(fmt(v, pct=pct) if not (key.endswith("_ms_mean")) else fmt(v, 0))
            rows.append([label] + cells)
        A(table(headers, rows))
        A("")
        A("说明：")
        A("")
        A("- **正确性/忠实度/完整性**由 LLM-as-judge 打 1~5 分。judge 拿到【问题】【标准答案】"
          "【检索到的证据】【模型回答】四样东西：正确性对着标准答案判，忠实度对着证据判"
          "（用来抓「超纲」和「编造」），完整性看标准答案的要点覆盖了多少。")
        A("- **拒答率**只在 8 道不可答题上统计。不可答题的正确行为是 `status != ok`；"
          "检索侧拒答（`refused`/`no_evidence`）和模型侧自报不足（`insufficient_evidence`）"
          "都算正确。")
        A("")

    # ---------------- 逐题明细 ----------------
    A("## 4. 逐题明细")
    A("")
    detail_key = "hybrid:rerank"
    rows = ret_rows.get(detail_key, [])
    rows_sorted = sorted(rows, key=lambda r: (r["status"] != "ok", r["mrr"], r["qid"]))
    A(f"### 4.1 检索失败与降级（{CONFIG_LABEL[detail_key]}）")
    A("")
    bad = [r for r in rows_sorted if r["status"] == "error" or r["n_relevant"] == 0
           or (by_qid[r["qid"]]["answerable"] and r["recall"] != 1.0)]
    if not bad:
        A("无：该配置下每道可答题都召回了全部 gold 文档。")
    else:
        A(table(["题目", "题型", "状态", "命中/相关", "gold 排名", "R@8", "题目"],
                [[r["qid"], by_qid[r["qid"]]["qtype"], r["status"],
                  f"{r['n_hits']}/{r['n_relevant']}", r["first_gold_rank"] or "未命中",
                  fmt(r["recall"], pct=True), by_qid[r["qid"]]["question"][:46]]
                 for r in bad]))
    A("")

    if gen_rows:
        A("### 4.2 不可答题的拒答行为")
        A("")
        gk = gkeys[0]
        rows = [r for r in gen_rows.get(gk, []) if not by_qid[r["qid"]]["answerable"]]
        A(table(["题目", "status", "是否拒答", "答案/拒答文案"],
                [[r["qid"], r["status"], "✅" if r["status"] != "ok" else "❌ 硬答了",
                  (r.get("answer") or "")[:60]] for r in rows]))
        A("")

        A("### 4.3 答案与引用抽样")
        A("")
        ok_rows = [r for r in gen_rows.get(gk, [])
                   if by_qid[r["qid"]]["answerable"] and r.get("status") == "ok"][:12]
        for r in ok_rows:
            it = by_qid[r["qid"]]
            j = r.get("judge") or {}
            A(f"**{r['qid']}**（{it['qtype']}）{it['question']}")
            A("")
            A(f"- 标准答案：{it['expected_answer']}")
            A(f"- 系统回答：{(r.get('answer') or '').strip()[:300]}")
            A(f"- 引用：{len(r.get('citations') or [])} 条，有效={r.get('citations_valid')}，"
              f"击中 gold={fmt(r.get('citation_doc_precision'), pct=True)}")
            A(f"- 评测：正确性={j.get('correctness')} 忠实度={j.get('faithfulness')} "
              f"完整性={j.get('completeness')} — {j.get('rationale', '')}")
            A("")

    return "\n".join(L)


INTENT = {
    "factoid": "词汇/向量召回的基本盘",
    "paraphrase": "换成同义表达，考语义召回",
    "numeric": "数字、日期、金额必须一字不差",
    "multi_hop": "需要跨两篇文档或同篇两处整合",
    "list": "答案是并列项，考完整率",
    "unanswerable": "库里没有，考「不该答就别答」",
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(WORK / "RAG评测报告.md"))
    args = ap.parse_args()
    text = build(args.tag)
    Path(args.out).write_text(text, encoding="utf-8")
    print("wrote", args.out, len(text), "chars")
