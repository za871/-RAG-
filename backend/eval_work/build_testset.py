"""从 candidates.json 组装最终测试集 testset.jsonl。

做三件事：
1. 把 evidence_spans 自动归属到承载它的 gold 文档（多跳题的 span 分属不同文档，
   靠字符串包含关系判定，比人工排顺序可靠）；
2. 若某个 span 在 gold 之外的文档里也逐字出现（跨文档的样板句，如省科技项目的
   年龄上限），把那篇文档一并纳入 gold —— 命中文档去回答同一个问题同样算对，
   不纳入会人为压低召回；
3. 补上人工编写的「不可答」题（这些主题在全库 0 命中，用于测拒答能力）。
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
cands = json.loads((HERE / "candidates.json").read_text(encoding="utf-8"))
index = json.loads((HERE / "corpus_index.json").read_text(encoding="utf-8"))
docs = {i: (HERE / f"docs/{i:03d}.txt").read_text(encoding="utf-8") for i in range(len(index))}

items = []
for c in cands:
    gold = list(c["gold_doc_indices"])
    # 1) 跨文档样板句 → 扩充 gold
    for span in c["evidence_spans"]:
        for i, text in docs.items():
            if span in text and i not in gold:
                gold.append(i)
    gold = sorted(set(gold))

    # 2) span → 所属 gold 文档
    evidence = []
    for span in c["evidence_spans"]:
        owners = [i for i in gold if span in docs[i]]
        evidence.append({"doc_index": owners[0] if owners else gold[0], "span": span})

    items.append({
        "qid": c["qid"],
        "question": c["question"],
        "qtype": c["qtype"],
        "difficulty": c["difficulty"],
        "answerable": True,
        "gold_doc_indices": gold,
        "gold_doc_ids": [index[i]["id"] for i in gold],
        "gold_filenames": [index[i]["filename"] for i in gold],
        "evidence": evidence,
        "expected_answer": c["expected_answer"],
    })

UNANSWERABLE = [
    ("u01", "学校教职工住房公积金的缴存比例是多少？学校和个人各承担多少？", "easy"),
    ("u02", "学校食堂的营业时间是几点到几点？周末会不会调整？", "easy"),
    ("u03", "图书馆的开馆时间和闭馆时间是几点？", "easy"),
    ("u04", "学生宿舍晚上几点熄灯？节假日熄灯时间会延后吗？", "easy"),
    ("u05", "学校校车（通勤班车）的发车时间和行驶路线是怎么安排的？", "medium"),
    ("u06", "学生浴室的开放时间和洗澡的收费标准是多少？", "medium"),
    ("u07", "学生参加城乡居民基本医疗保险的缴费标准和门诊报销流程是什么？", "hard"),
    ("u08", "推荐优秀应届本科毕业生免试攻读研究生（推免）的名额和申请条件是什么？", "hard"),
]
for qid, q, diff in UNANSWERABLE:
    items.append({
        "qid": qid,
        "question": q,
        "qtype": "unanswerable",
        "difficulty": diff,
        "answerable": False,
        "gold_doc_indices": [],
        "gold_doc_ids": [],
        "gold_filenames": [],
        "evidence": [],
        "expected_answer": None,
    })

out = HERE / "testset.jsonl"
with out.open("w", encoding="utf-8") as fh:
    for item in items:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")

from collections import Counter
print("items:", len(items))
print("by type:", Counter(i["qtype"] for i in items).most_common())
print("by difficulty:", Counter(i["difficulty"] for i in items).most_common())
print("multi-gold questions:", [(i["qid"], i["gold_doc_indices"]) for i in items if len(i["gold_doc_indices"]) > 1])
print("wrote", out)
