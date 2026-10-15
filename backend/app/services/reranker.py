"""本地 cross-encoder 重排模型（BAAI/bge-reranker-v2-m3），给召回结果重算「query-段落」相关性。

为什么需要它：RRF 只用名次融合两路召回，**从不重算相关性**——一个 chunk 在 BM25 里名次高、
但它和问题的语义其实不搭，RRF 也照样把它排前面。cross-encoder 把 (query, 正文) 拼成
一个序列一起过模型，能直接给「这一对有多相关」打分，比双塔余弦 / 词频都准。

为什么用 sigmoid 归一化而不是 softmax / min-max：
  - sigmoid 是**逐对**计算的，分数只取决于这一对 (query, chunk)，与候选集合无关
    → 同一个 chunk 在不同 top_k、不同候选池下分数不变，跨查询可比，阈值才有意义；
  - softmax / min-max 依赖候选集合，候选一换分数就变，阈值无从固定。
  代价是分数不再「加和为 1」，本来也不需要。

模型解析规则与 embedding 一致（见 config.resolved_reranker_model）：
优先 backend/models/<name>，否则当 HuggingFace repo id 交给本地缓存处理。
local_files_only=True：本机网络受限，且权重已完整缓存，不做任何在线元数据检查。
"""

from functools import lru_cache
from typing import Any, Mapping, Sequence

import torch
from sentence_transformers import CrossEncoder

from app.config import get_settings

# 子块最长 420 字（chunking.child_size），远不到这个上限；它只是用来兜住超长的 query
MAX_LENGTH = 1024


@lru_cache
def get_reranker() -> CrossEncoder:
    """进程内缓存的 cross-encoder。首次调用要加载 ~2.2G 权重（冷启动，见 README）。"""
    settings = get_settings()
    try:
        return CrossEncoder(
            model_name_or_path=settings.resolved_reranker_model,
            # 复用 embedding 的 device：device 是机器的属性、不是模型的属性，
            # 单开一个 RERANKER_DEVICE 只会招来「embedding 在 cuda、重排在 cpu」这种错配
            device=settings.embedding_device,
            # 显式指定 sigmoid → predict() 返回的就已经是 0~1，外层不再做任何归一化。
            # 必须放在构造函数上而不是手动 torch.sigmoid(predict(...))：手动会二次压缩到 0.5 附近
            activation_fn=torch.nn.Sigmoid(),
            max_length=MAX_LENGTH,
            local_files_only=True,
        )
    except Exception as exc:
        # 服务端配置问题（权重没缓存 / 路径写错），不是用户输入问题
        raise RuntimeError(
            f"重排模型 {settings.resolved_reranker_model} 无法加载：{exc}。"
            f"请确认它已在本地缓存，或把它放到 backend/models/{settings.reranker_model} 下。"
        ) from exc


def rerank(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    min_score: float,
) -> list[tuple[str, float]]:
    """按 query-candidate 相关性重排 + 阈值过滤，返回 (chunk_id, rerank_score) 降序列表。

    rerank_score = sigmoid(交叉编码器 logit)，落在开区间 (0,1)、越高越相关；
    逐对计算 ⇒ 与候选池大小、top_k 都无关 ⇒ 跨查询可比，阈值才有意义。

    刻意做成纯函数（不读配置、不碰 DB / Milvus，top_k / min_score 由调用方传入），
    这样可以直接拿真实模型单测，不影响检索管线。

    **先阈值过滤、再按 top_k 截断**，顺序不能反：反过来 top_k 会把「本来过线但排在后面」
    的证据挡掉，阈值就失去了「判证据够不够」的意义。

    候选为空直接返回 []：**绝不触发模型加载**——2.2G 权重不该为「无候选」买单。
    """
    if not candidates:
        return []

    pairs = [(query, str(candidate.get("content") or "")) for candidate in candidates]
    # num_labels == 1：predict 返回 shape (n,) 的 float32 ndarray，
    # 与 pairs 严格同序（内部按长度分批再还原，顺序对调用方无感）
    scores = get_reranker().predict(
        pairs,
        # 候选池本来就被 pool_k 封在个位数，一次前向就够，不必再引入一个 batch_size 旋钮
        batch_size=len(pairs),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    kept = [
        (str(candidate["chunk_id"]), float(score))
        for candidate, score in zip(candidates, scores)
        if float(score) >= min_score
    ]
    # 分降序；同分按 chunk_id 升序 → 名次完全确定，与候选插入顺序无关
    # （和 bm25_store.search_children_bm25 同一套约定）
    kept.sort(key=lambda item: (-item[1], item[0]))
    return kept[:top_k]
