"""在一些数据库中，schema就是数据库的意思。
    在一些代码编程的技术中，schema是表结构的意思"""
from datetime import date,datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
class ORMModel(APIModel):
    model_config=ConfigDict(from_attributes=True,extra="forbid")
#将上传文件的结果封装到这个实体中，返回给界面使用
class DocumentOut(ORMModel):
    id:str
    filename:str
    content_type:str
    topic:str | None
    effective_date:date | None
    status:str
    error_message:str | None
    chunk_count:int
    index_version:int
    created_at:datetime
#分页查询的返回结构：items是当前页的数据，total是符合筛选条件的总条数
class DocumentPage(APIModel):
    items:list[DocumentOut]
    total:int
    page:int
    size:int

#纯文本入库的入参：不走文件上传，正文直接提交进来切片
class TextIngestRequest(APIModel):
    #对应 PolicyDocument.filename 列（String(225)），MySQL 严格模式下超长会直接报错，这里先卡住
    title:str | None = Field(default=None,max_length=225)
    content:str = Field(min_length=1)
    topic:str | None = None
    effective_date:date | None = None

#检索的入参（向量 / BM25 / 混合融合）
class SearchRequest(APIModel):
    query:str = Field(min_length=1)
    top_k:int = Field(default=8,ge=1,le=50)
    #只在这个文档范围内检索；不传就是全库
    document_id:str | None = None
    #只在这个主题内检索；对齐 MySQL 侧 topic 列的 String(100)
    topic:str | None = Field(default=None,max_length=100)
    #只取生效日期在这一天（含）之前的政策；不传就不按日期筛
    effective_before:date | None = None
    #检索模式：vector=纯向量（和加 BM25 之前的行为完全一致）；bm25=纯关键词；
    #hybrid=两路各取候选后用 RRF 融合（默认）。用 Literal 让 FastAPI 直接校验成 422
    mode:Literal["vector","bm25","hybrid"] = "hybrid"
    #是否对召回结果做 cross-encoder 重排。省略（None）则跟随服务端的 ENABLE_RERANKER 配置；
    #显式传 false 可以在单次请求里关掉它、拿到「重排前」的原始顺序
    rerank:bool | None = None

#检索命中的单条子块
class SearchHit(APIModel):
    chunk_id:str
    document_id:str
    parent_id:str | None
    content:str
    section:str | None
    page:int | None
    ordinal:int
    #COSINE 度量下这是相似度，越高越相关。
    #BM25-only 命中的子块没有向量相似度，所以这里可以是 None
    distance:float | None = None
    #BM25 原始分，越高越相关；没参与 BM25 那一路时是 None
    bm25_score:float | None = None
    #RRF 融合分，越高越靠前；只在 hybrid 模式下才有，且只在同一次响应内可比
    rrf_score:float | None = None
    #cross-encoder 重排分 = sigmoid(原始 logit)，落在 (0,1)、越高越相关。
    #逐对计算所以跨查询可比；重排关闭时为 None，开启时它就是排序依据
    rerank_score:float | None = None
    #回显命中块所属文档的主题与生效日期（当初没填的为 None）
    topic:str | None
    effective_date:date | None
    #按 small-to-big 从 MySQL 回查回来的父块正文，给 LLM 当上下文
    parent_content:str | None

class SearchResponse(APIModel):
    query:str
    top_k:int
    #回显实际生效的检索模式，方便排查「为什么这次没走 BM25」
    mode:str
    #ok=有证据；no_evidence=召回本身就是空（库空/过滤全不中/分词为空）；
    #refused=召回到了但全部低于 MIN_EVIDENCE_SCORE。见 document_service 的说明
    status:Literal["ok","no_evidence","refused"] = "ok"
    #拒答原因，给人看的；status 为 ok 时必须是 None
    reason:str | None = None
    hits:list[SearchHit]

    @model_validator(mode="after")
    def _refuse_implies_empty(self):
        """把「非 ok 就没有证据」这个不变量写死在模型上，而不是只靠调用方自觉。"""
        if self.status != "ok" and self.hits:
            raise ValueError(f"status={self.status} 时 hits 必须为空")
        return self

#RAG 问答的入参：检索过滤条件与 SearchRequest 逐一对齐。
#问答的召回范围必须和 /search 完全同构，否则「同一句话在 /search 能召回、在 /qa/ask 不能」
#会变成排查负担。没有 history/session_id：这个接口是一轮一问、没有会话概念
class AskRequest(APIModel):
    query:str = Field(min_length=1)
    top_k:int = Field(default=8,ge=1,le=50)
    document_id:str | None = None
    topic:str | None = Field(default=None,max_length=100)
    effective_before:date | None = None
    mode:Literal["vector","bm25","hybrid"] = "hybrid"
    rerank:bool | None = None

#一条被答案真正引用到的证据
class Citation(APIModel):
    #引用号 = 送进 prompt 的证据编号（【证据 n】里的 n）
    index:int
    #代表块：该父块下排名最高的那个子块
    chunk_id:str
    document_id:str
    filename:str
    section:str | None
    page:int | None
    topic:str | None
    effective_date:date | None
    #证据正文前 120 字，方便前端直接展示「依据是什么」
    snippet:str
    rerank_score:float | None = None

class Timings(APIModel):
    #检索总耗时；重排在 search_documents 内部，所以并在这里，不单列
    retrieval_ms:float
    #大模型调用耗时；拒答路径不调模型，恒为 0
    llm_ms:float
    #端到端，含证据组装、filename 回查、引用解析
    total_ms:float

#实际发给大模型的消息，逐字回放
class ChatMessage(APIModel):
    role:str
    content:str

class EvidenceScore(APIModel):
    index:int
    chunk_id:str
    rerank_score:float | None

class DebugInfo(APIModel):
    model:str
    temperature:float
    #实际生效的重排开关（None 已在服务层解析成具体 bool，所以这里不会含糊）
    rerank:bool
    retrieval_status:str
    retrieval_reason:str | None
    #去重前的命中数
    hit_count:int
    #按父块折叠 + 预算截断后的证据条数
    evidence_count:int
    #因上下文预算被整条丢掉的证据数
    evidence_dropped:int
    evidence_chars:int
    evidence_scores:list[EvidenceScore]
    llm_called:bool
    #system + user 的完整原文
    messages:list[ChatMessage]
    #大模型回的 usage_metadata（token 用量）
    usage:dict[str,Any]
    cited_indices:list[int]
    invalid_citation_indices:list[int]
    #模型是否回了 INSUFFICIENT_EVIDENCE 标记
    llm_insufficient:bool
    #模型原始输出（角标剔除前）；只有真调了模型才有值
    answer_raw:str | None

class AskResponse(APIModel):
    query:str
    #no_evidence=库里没有；refused=召回不够相关（检索侧判定）；
    #insufficient_evidence=材料不足以回答（大模型侧判定）。前两个原样透传检索状态
    status:Literal["ok","no_evidence","refused","insufficient_evidence"] = "ok"
    reason:str | None = None
    answer:str
    citations:list[Citation]
    #只对 status=ok 有意义：ok 时要求「至少有一条引用」且「没有越界引用」
    citations_valid:bool = True
    timings:Timings
    debug:DebugInfo

    @model_validator(mode="after")
    def _refuse_implies_no_citations(self):
        """非 ok 就没有任何答案断言，因此不可能有引用。与 SearchResponse 同一套思路。"""
        if self.status != "ok" and self.citations:
            raise ValueError(f"status={self.status} 时 citations 必须为空")
        return self

#删除单个文档的结果
class DeleteResponse(APIModel):
    deleted:bool
    document_id:str

#重置（清空全部）的结果
class ResetResponse(APIModel):
    documents_deleted:int
    chunks_deleted:int
    files_deleted:int