"""本地 BGE embedding 模型（sentence-transformers），替代原 DashScope 云端 embedding。"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

# BGE 中文模型的官方检索指令前缀：只加在 query 上，文档不加。
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.resolved_embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={
            "batch_size": settings.embedding_batch_size,
            "normalize_embeddings": True,   # 余弦相似度检索，显式归一化
        },
        # 非空时 embed_query() 走这套参数，embed_documents() 不受影响
        query_encode_kwargs={"prompt": BGE_ZH_QUERY_INSTRUCTION},
      )
