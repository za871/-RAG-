"""Milvus 读写封装。

只往 Milvus 里写**子块**（向量 + 少量标量字段），父块留在 MySQL，
检索命中子块后用 parent_id 回查父块正文做上下文扩展（small-to-big）。
"""

from datetime import date
from functools import lru_cache
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.config import get_settings

# 子块最长 420 字（chunking.child_size），8192 留足余量
CONTENT_MAX_LEN = 8192
SECTION_MAX_LEN = 512
ID_MAX_LEN = 36

# topic 的 VARCHAR 长度是按 **UTF-8 字节**算的（中文 1 个字 3 字节），不是字符数。
# MySQL 侧 topic 是 String(100) 字符、最多 300 字节，所以这里留足余量取 512，
# 照抄 100 会导致长中文主题在插入时才被服务端拒绝
TOPIC_MAX_LEN = 512

# Milvus 的标量字段不支持 null，docx/txt 没有页码时用 -1 占位
NO_PAGE = -1
# 生效日期未填时的哨兵值，存成 YYYYMMDD 的 int，字典序即时间序。
# 取 99991231 而不是 0：这样 `effective_date <= X` 会把「没填生效日期」的文档排除掉——
# 日期未知时无从判断它是否在 X 之前生效，宁可漏也不要错引一份政策
DATE_MAX = 99991231

# schema 必须齐全的字段名。老 collection 缺了 topic / effective_date，
# 过滤表达式一上去就会被判「未知字段」，所以在 ensure_collection 里显式拦住
REQUIRED_FIELDS = {
    "chunk_id", "vector", "document_id", "parent_id",
    "content", "section", "page", "ordinal",
    "topic", "effective_date",
}


def truncate_utf8(value: str, max_bytes: int) -> str:
    """按 UTF-8 **字节**截断，且不切碎多字节字符。

    Milvus 的 VARCHAR max_length 是按字节算的（中文 1 字 3 字节），超了要等插入时
    才由服务端报 `length of varchar field ... exceeds max length`——那时整条文档都已经
    切完块、跑完 embedding 了，白烧一次 CPU。所以入库前主动截断。
    errors="ignore" 让恰好被切开的那个字符丢掉而不是变成乱码。
    """
    raw = value.encode("utf-8")
    return value if len(raw) <= max_bytes else raw[:max_bytes].decode("utf-8", errors="ignore")


def effective_date_to_int(value: date | None) -> int:
    """生效日期转成 YYYYMMDD 的 int（字典序即时间序）；没填就用 DATE_MAX 这个哨兵。"""
    return int(value.strftime("%Y%m%d")) if value else DATE_MAX


def date_from_int(value: int | None) -> date | None:
    """把 Milvus 里的 YYYYMMDD 还原成 date；哨兵值（当初没填）还原成 None。"""
    if not value or value == DATE_MAX:
        return None
    text = str(value)
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _build_schema(dim: int) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            # 主键就是 MySQL document_chunks.id，两边靠它对齐，不再生成第二套 id
            FieldSchema("chunk_id", DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LEN),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("document_id", DataType.VARCHAR, max_length=ID_MAX_LEN),
            FieldSchema("parent_id", DataType.VARCHAR, max_length=ID_MAX_LEN),
            FieldSchema("content", DataType.VARCHAR, max_length=CONTENT_MAX_LEN),
            FieldSchema("section", DataType.VARCHAR, max_length=SECTION_MAX_LEN),
            FieldSchema("page", DataType.INT64),
            FieldSchema("ordinal", DataType.INT64),
            # 主题和生效日期本来只在 MySQL 的 policy_documents 上，
            # 这里冗余一份，好让「限定主题 / 限定某日期之前生效」也能在 Milvus 里原生过滤
            FieldSchema("topic", DataType.VARCHAR, max_length=TOPIC_MAX_LEN),
            FieldSchema("effective_date", DataType.INT64),
        ]
    )


def _assert_schema_current(client: MilvusClient, name: str) -> None:
    """老集合缺了新列时直接报错，而不是放任它跑到「未知字段」那种难懂的报错。

    尤其这是检索路径：静默少召回（比如过滤条件失效、结果悄悄变多或变少）
    比直接报错危险得多，所以这里宁可炸掉
    """
    present = {field["name"] for field in client.describe_collection(name)["fields"]}
    missing = REQUIRED_FIELDS - present
    if missing:
        raise RuntimeError(
            f"Milvus collection {name!r} 缺少字段 {sorted(missing)}，是旧版 schema。"
            "请先跑 python scripts/reindex_milvus.py 从 MySQL 重建向量库。"
        )


@lru_cache
def get_milvus_client() -> MilvusClient:
    settings = get_settings()
    kwargs: dict[str, Any] = {"uri": settings.milvus_uri}
    # 本地 standalone 不带鉴权，token 是空串；传空串反而可能被当成非法凭证
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def ensure_collection(client: MilvusClient) -> str:
    """collection 不存在就建 schema + 索引并 load；已存在则校验过 schema 后原样返回名字。"""
    settings = get_settings()
    name = settings.milvus_collection
    if client.has_collection(name):
        _assert_schema_current(client, name)
        return name

    index_params = client.prepare_index_params()
    # 向量已做 L2 归一化，所以用 COSINE；HNSW 在中小规模上召回和延迟比较均衡
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # 标量索引：让过滤表达式能走索引，而不是逐个 segment 全量比对。
    # 这三种类型（INVERTED / STL_SORT）已在 Milvus 3.0.2 上实测通过
    # 等值过滤（topic / document_id）用 INVERTED
    index_params.add_index(field_name="topic", index_type="INVERTED")
    index_params.add_index(field_name="document_id", index_type="INVERTED")
    # 数值范围过滤（effective_date <= X）用 STL_SORT
    index_params.add_index(field_name="effective_date", index_type="STL_SORT")
    client.create_collection(
        collection_name=name,
        schema=_build_schema(settings.embedding_dim),
        index_params=index_params,
    )
    client.load_collection(name)
    return name


def drop_collection(client: MilvusClient) -> None:
    """整个删掉 collection，用于重置：向量、索引、segment 一次清干净。

    下次入库时 upsert_child_chunks 会调 ensure_collection 按 schema 重新建好，
    所以 drop 之后不需要任何补救代码。collection 不存在时静默返回，天然幂等。
    """
    name = get_settings().milvus_collection
    if client.has_collection(name):
        client.drop_collection(name)


def delete_document(client: MilvusClient, document_id: str) -> None:
    """删掉某文档的所有子块向量。collection 不存在时静默返回。"""
    name = get_settings().milvus_collection
    if not client.has_collection(name):
        return
    client.delete(collection_name=name, filter=f'document_id == "{document_id}"')
    # 必须 flush，和 upsert_child_chunks 里那句同一个道理：
    # 删除只是写了一条墓碑记录，不 flush 的话随后的 search 仍会把已删的子块检索出来
    # （实测删完立刻检索还能命中，flush 之后才消失）。删文档后用户马上再问一句是很正常的用法，
    # 不能指望恰好有别的写入帮我们 flush
    client.flush(name)


def upsert_child_chunks(
    client: MilvusClient,
    document_id: str,
    rows: list[dict[str, Any]],
) -> int:
    """写入某文档的子块。先按 document_id 删旧再插新，实现幂等 upsert。

    rows 的键：chunk_id / vector / document_id / parent_id / content / section / page / ordinal
    返回实际插入条数。
    """
    ensure_collection(client)
    # 幂等的关键：重新摄取同一文档时不会留下上一轮的残留向量
    delete_document(client, document_id)
    if not rows:
        return 0

    name = get_settings().milvus_collection
    # 三个 VARCHAR 字段都按各自的字节上限截断后再插。
    # 放在这一层而不是各个调用方：schema 的长度约束是这里定义的，就该由这里兜住，
    # 否则 reindex_milvus.py 之类的旁路调用方会重复踩同一个坑
    rows = [
        {
            **row,
            "content": truncate_utf8(row.get("content") or "", CONTENT_MAX_LEN),
            "section": truncate_utf8(row.get("section") or "", SECTION_MAX_LEN),
            "topic": truncate_utf8(row.get("topic") or "", TOPIC_MAX_LEN),
        }
        for row in rows
    ]
    result = client.insert(collection_name=name, data=rows)
    # flush 让数据对随后的 query/search 立即可见（否则可能落在 growing segment 里查不到）
    client.flush(name)
    return int(result.get("insert_count", 0))


def search_children(
    client: MilvusClient,
    query_vector: list[float],
    top_k: int = 8,
    filter_expr: str = "",
) -> list[dict[str, Any]]:
    """按向量检索子块，返回带标量字段的普通 dict 列表（含 distance）。

    collection 不存在直接返回空列表：全新的库或刚重置过的库「什么都没索引」，
    属于正常的 200 + 空结果，不该报错。这里刻意不调 ensure_collection——
    检索是只读操作，不该顺手建集合。

    distance 语义由 metric_type 决定：COSINE 下它就是相似度，**越高越相关**，
    且 Milvus 已按最优在前排好序，调用方不要再重排。
    """
    name = get_settings().milvus_collection
    if not client.has_collection(name):
        return []

    hits = client.search(
        collection_name=name,
        data=[query_vector],
        filter=filter_expr,
        limit=top_k,
        output_fields=[
            "chunk_id", "document_id", "parent_id", "content",
            "section", "page", "ordinal", "topic", "effective_date",
        ],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )
    # MilvusClient.search 返回的是嵌套结构：[{"chunk_id":…, "distance":…, "entity":{标量字段}}]，
    # 标量字段在 entity 里。这里拍平成一层，调用方就不必碰 pymilvus 的内部结构
    rows = hits[0] if hits else []
    return [
        {
            "chunk_id": entity.get("chunk_id"),
            "document_id": entity.get("document_id"),
            "parent_id": entity.get("parent_id") or "",
            "content": entity.get("content") or "",
            "section": entity.get("section") or "",
            "page": entity.get("page"),
            "ordinal": entity.get("ordinal"),
            "topic": entity.get("topic") or "",
            "effective_date": entity.get("effective_date"),
            "distance": hit.get("distance"),
        }
        for hit in rows
        for entity in (hit.get("entity") or {},)
    ]
