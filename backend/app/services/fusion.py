"""RRF（Reciprocal Rank Fusion）融合。

把多个检索器返回的「已按相关性降序排好的 id 列表」合成一个有序列表。
**只用名次、不用分数**——这正是要它的原因：余弦相似度和 BM25 分数量纲完全不同，
直接加权求和既需要调参、又不可迁移；RRF 免调参、对两者的分数尺度免疫。

放在单独模块而不是塞进 document_service：它不认识任何检索器，
只吃「有序 id 列表」，将来加第三路（比如标题匹配）不用改它。
"""

from collections.abc import Sequence

# 标准 RRF 的经验值：k=60（Cormack et al. 2009）。k 越大越弱化「第一名」的独占优势，
# 让深位置的结果也有机会。本场景候选池只有几十条，60 是稳妥默认值，不是调优结果。
RRF_K = 60

# 融合前每个检索器各自多取一些，给「另一个检索器也命中」的块留出加分空间。
# top_k=8 → 32：并集够大，又不至于让两路检索都变慢。
CANDIDATE_K_MULTIPLIER = 4
CANDIDATE_K_MIN = 20
CANDIDATE_K_MAX = 100


def fusion_candidate_k(top_k: int) -> int:
    """融合时每个检索器取多少候选（上下限都卡住，防止 top_k 很大时候选池失控）。"""
    return min(max(top_k * CANDIDATE_K_MULTIPLIER, CANDIDATE_K_MIN), CANDIDATE_K_MAX)


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """融合多个「已按相关性降序排好的 id 列表」，返回 (id, rrf_score) 降序列表。

    公式：score(d) = Σ_r 1 / (k + rank_r(d))，rank 从 1 开始。
    各列表长度可以不同；同一个 id 出现多次就分数相加——
    「两个检索器都看中」本来就比「只有一路看中」更可信。

    确定性（RRF 的并列很常见，必须可复现）：
      - dict 保插入序，插入序就是「首次出现的顺序」；
      - sorted 是稳定排序，等分项保持插入序。
    两条合起来 = 同分时「先出现的排前面」，也就是 rankings[0] 的次序起主导。
    全程不碰 set / random，同样输入必然同样输出。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda pair: -pair[1])
