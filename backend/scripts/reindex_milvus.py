"""从 MySQL 重建整个 Milvus 向量库。

两个用途：
  1. Milvus schema 升级后的数据迁移。例如新增 topic / effective_date 两列后，
     老 collection 缺列，ensure_collection 会直接报错，跑一次本脚本即可补上。
  2. 修复 ingest 半失败留下的「MySQL 有块、Milvus 没有」状态，
     顺便把 status 修正回 ready。

为什么不需要原始上传文件：document_chunks 表里父块子块的正文都在，
MySQL 是元数据的真相来源，拿切块重新 embedding 就能把向量库整个重建出来。

    python scripts/reindex_milvus.py
"""

import sys
from pathlib import Path

# 直接 `python scripts/reindex_milvus.py` 时 sys.path[0] 是 scripts/ 而不是 backend/，
# import app 会 ModuleNotFoundError；把 backend/ 补进来，从哪个目录执行都一样
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.database import SessionLocal
from app.models import DocumentChunk, DocumentStatus, PolicyDocument
from app.services.milvus_store import (
    NO_PAGE,
    drop_collection,
    effective_date_to_int,
    get_milvus_client,
    upsert_child_chunks,
)
from app.services.vector_store import get_embeddings


def main():
    client = get_milvus_client()
    # 老 collection 的 schema 缺列，直接 drop 从头建。数据从 MySQL 重灌，不怕丢
    drop_collection(client)
    print("已删除旧 collection，开始从 MySQL 重建……")

    embeddings = get_embeddings()
    document_count = 0
    total_chunks = 0

    with SessionLocal() as session:
        documents = session.scalars(
            select(PolicyDocument).order_by(PolicyDocument.created_at)
        ).all()

        for document in documents:
            chunks = session.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == document.id,
                    # 只有子块进 Milvus，父块留在 MySQL 供回查
                    DocumentChunk.is_parent.is_(False),
                )
                .order_by(DocumentChunk.ordinal)
            ).all()
            if not chunks:
                print(f"  跳过 {document.filename}（没有子块）")
                continue

            vectors = embeddings.embed_documents([chunk.content or "" for chunk in chunks])
            count = upsert_child_chunks(
                client,
                document.id,
                [
                    {
                        # 字段口径必须和 document_service._chunk_persist_and_index 保持一致
                        "chunk_id": chunk.id,
                        "vector": vector,
                        "document_id": document.id,
                        "parent_id": chunk.parent_id or "",
                        "content": chunk.content or "",
                        "section": chunk.section or "",
                        # Milvus 标量字段不支持 null，没有页码时用 -1 占位
                        "page": chunk.metadata_json.get("page")
                        if chunk.metadata_json.get("page") is not None
                        else NO_PAGE,
                        "ordinal": chunk.ordinal,
                        "topic": document.topic or "",
                        "effective_date": effective_date_to_int(document.effective_date),
                    }
                    for chunk, vector in zip(chunks, vectors)
                ],
            )

            # 顺便把状态修正回 ready：半失败留下的 failed 也一起修好
            document.chunk_count = count
            document.status = DocumentStatus.ready
            document.index_version += 1
            document.error_message = None
            document_count += 1
            total_chunks += count
            print(f"  {document.filename}: {count} 条")

        session.commit()

    print(f"重建完成：{document_count} 个文档，{total_chunks} 条子块")


if __name__ == "__main__":
    main()
