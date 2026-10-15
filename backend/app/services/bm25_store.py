"""内存态 BM25 索引（rank_bm25），语料来自 MySQL 的子块。

只索引子块（is_parent=False），和 Milvus 保持同一批 chunk_id，两路检索才对得上。
topic / effective_date 不进 Milvus 也不在 chunk 表上，装载时从 policy_documents
JOIN 下来冗余进内存，检索时过滤就是纯 Python 比较——不用回表，也不用建第二套索引。

进程内缓存，内容变化（入库 / 删除 / 重置）时由 document_service 显式调
invalidate_bm25_index() 失效。**多 worker 下不共享**：worker A 的写入不会失效
worker B 的缓存。当前是单进程部署，够用；真要上多 worker 得再加版本戳或 TTL。
"""

from dataclasses import dataclass
from datetime import date
import logging
from functools import lru_cache
import re
import threading

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DocumentChunk, DocumentStatus, PolicyDocument
from app.services.milvus_store import NO_PAGE, effective_date_to_int

# 判据是「至少含一个中文/字母/数字字符」：中文词、英文词、数字都留，
# 纯标点（，。！？（））和空白丢掉——它们对 BM25 只有噪声
_TOKEN_HAS_ALNUM = re.compile(r"[0-9A-Za-z一-鿿]")


@lru_cache
def _ensure_jieba() -> None:
    """jieba 首次分词要构建前缀词典（约 1 秒）并打一堆 DEBUG 日志，这里只做一次。

    不在 import 时初始化：纯向量模式和不碰 BM25 的脚本不该为它付启动代价。
    """
    jieba.setLogLevel(logging.WARNING)
    jieba.initialize()


def tokenize(text: str | None) -> list[str]:
    """把中英混排文本切成 BM25 token。语料侧和 query 侧必须调同一个函数。

    - 中文走 jieba 精确模式（lcut）。不用 cut_for_search：它会把长词再拆成子词，
      在 420 字的短子块上会虚高召回、还把 IDF 稀释了。
    - 英文/数字 jieba 原样吐出整个 token，统一 lower，让 GDP 能命中 gdp。
    - 标点/空白丢掉。
    """
    if not text:
        return []
    _ensure_jieba()
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if token and _TOKEN_HAS_ALNUM.search(token):
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class Bm25Entry:
    """一个子块在内存里的全部标量字段——对齐 milvus_store 的哨兵约定，
    好让 _materialize_hits 可以用同一套还原逻辑处理两路命中。"""

    chunk_id: str
    document_id: str
    parent_id: str      # "" = 无父块（对齐 Milvus 的空串哨兵）
    content: str
    section: str        # "" = 无章节
    page: int           # 无页码 = NO_PAGE(-1)
    ordinal: int
    topic: str          # "" = 未填主题
    effective_date: int  # YYYYMMDD；未填 = DATE_MAX(99991231)


@dataclass(frozen=True)
class Bm25Index:
    # 语料为空时是 None：BM25Okapi([]) 会除零，而且空语料本来就无从打分
    bm25: BM25Okapi | None
    # 顺序与 bm25 的语料顺序严格一一对应（get_scores 的返回值按语料序）
    entries: tuple[Bm25Entry, ...]


_INDEX: Bm25Index | None = None
_LOCK = threading.Lock()


def invalidate_bm25_index() -> None:
    """唯一的失效入口。写路径（入库 / 删除 / 重置）**commit 之后**调用。

    必须在 commit 之后：否则落在「失效」与「commit」之间的那次检索，
    会把当时还没提交的语料建进索引，而那份缓存再也没有人来失效它。
    """
    global _INDEX
    with _LOCK:
        _INDEX = None


def get_bm25_index(session: Session) -> Bm25Index:
    """取进程内缓存的 BM25 索引，没建过就现建。

    双重检查加锁：冷启动时并发进来的多个请求只建一次，
    其余的在锁上等（全量装载 + 分词是在请求线程里同步做的，
    政策量级几千块无所谓；到十万块量级再挪到后台任务）。
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is None:
            _INDEX = _build_index(session)
        return _INDEX


def _build_index(session: Session) -> Bm25Index:
    """从 MySQL 装载全部子块 + 所属文档的 topic/effective_date，建 BM25Okapi。

    只收 status=ready 的文档，和向量路径对齐：向量路只能检索到成功写进 Milvus 的块
    （失败文档的向量已被 delete_document 清掉，indexing 中的文档还没写 Milvus），
    这里不过滤就会多召，两路的命中集合就对不上了。
    """
    rows = session.execute(
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.parent_id,
            DocumentChunk.content,
            DocumentChunk.section,
            DocumentChunk.ordinal,
            DocumentChunk.metadata_json,
            PolicyDocument.topic,
            PolicyDocument.effective_date,
        )
        .join(PolicyDocument, DocumentChunk.document_id == PolicyDocument.id)
        .where(
            DocumentChunk.is_parent.is_(False),
            PolicyDocument.status == DocumentStatus.ready,
        )
        # 按主键排序（唯一）→ 语料顺序完全确定，RRF 的并列名次不会因插入顺序抖动
        .order_by(DocumentChunk.id)
    ).all()

    entries: list[Bm25Entry] = []
    corpus: list[list[str]] = []
    for chunk_id, doc_id, parent_id, content, section, ordinal, meta, topic, eff_date in rows:
        page = (meta or {}).get("page")
        entries.append(Bm25Entry(
            chunk_id=chunk_id,
            document_id=doc_id,
            parent_id=parent_id or "",
            content=content or "",
            section=section or "",
            page=NO_PAGE if page is None else page,
            ordinal=ordinal,
            topic=topic or "",
            # 复用 milvus_store 的转换：两路对「没填生效日期」的理解天然一致
            effective_date=effective_date_to_int(eff_date),
        ))
        corpus.append(tokenize(content))

    if not corpus:
        return Bm25Index(bm25=None, entries=())
    return Bm25Index(bm25=BM25Okapi(corpus), entries=tuple(entries))


def _matches(
    entry: Bm25Entry,
    document_id: str | None,
    topic: str | None,
    effective_before: date | None,
) -> bool:
    """在内存里复刻 document_service 拼给 Milvus 的那套过滤表达式，逐条对齐。

    日期那条是重点：Milvus 侧是 `effective_date <= effective_date_to_int(X)`，
    这里用**同一个函数**转换后比大小。未填日期的文档存的是 DATE_MAX(99991231)，
    会被「<= X」自然排除——两条路的漏召/多召行为必须一模一样，
    否则同一个查询在不同 mode 下返回的命中集合会不一致。
    """
    if document_id and entry.document_id != document_id:
        return False
    if topic and entry.topic != topic:
        return False
    if effective_before is not None and entry.effective_date > effective_date_to_int(effective_before):
        return False
    return True


def search_children_bm25(
    index: Bm25Index,
    query: str,
    *,
    top_k: int = 8,
    document_id: str | None = None,
    topic: str | None = None,
    effective_before: date | None = None,
) -> list[dict]:
    """按 BM25 检索子块，返回按分数降序的扁平 dict 列表（含 bm25_score）。

    一个全局索引 + 查询时按谓词掩码，而不是每个过滤组合重建索引：
    rank_bm25 的 IDF 是**语料级**统计，重建只会让「同一句话在不同过滤下分数不一样」，
    反而更难解释；全局 IDF 在过滤子集上打分是本场景的标准做法，也便宜得多
    （get_scores 本来就是一次 numpy 全语料扫描，掩码筛选搭个便车）。
    """
    tokens = tokenize(query)
    if not tokens or index.bm25 is None:
        return []

    scores = index.bm25.get_scores(tokens)  # ndarray，与 index.entries 同序
    query_tokens = set(tokens)
    candidates = [
        (float(score), entry)
        # 用「共享至少一个 token」判命中，**不能**用 score > 0：
        # rank_bm25 对 idf 为负的词用 epsilon * average_idf 兜底，而当 query 词
        # 几乎出现在所有文档里（语料小、或是个泛词）时 average_idf 本身为负，
        # 分数就会是负的——那是真命中，用 > 0 会整批误杀。
        # doc_freqs[i] 是该文档的词频 Counter，和 entries 同序，直接复用不额外占内存
        for score, entry, doc_counter in zip(scores, index.entries, index.bm25.doc_freqs)
        if query_tokens & doc_counter.keys()
        and _matches(entry, document_id, topic, effective_before)
    ]
    # 分数降序；同分按 chunk_id 升序 → 名次完全确定，RRF 可复现
    candidates.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))

    return [
        {
            "chunk_id": entry.chunk_id,
            "document_id": entry.document_id,
            "parent_id": entry.parent_id,
            "content": entry.content,
            "section": entry.section,
            "page": entry.page,
            "ordinal": entry.ordinal,
            "topic": entry.topic,
            "effective_date": entry.effective_date,
            "bm25_score": score,
        }
        for score, entry in candidates[:top_k]
    ]
