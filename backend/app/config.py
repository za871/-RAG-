from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# backend/ 目录，用来把相对路径锚定成绝对路径
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class  Settings(BaseSettings):
    app_name:str=""
    debug:bool=True
    api_prefix:str="/api"   # 不能带尾斜杠，否则 include_router 会抛 AssertionError

    database_url:str=(
        "mysql+pymysql://root:123456@127.0.0.1:3306/gov_rag?charset=utf8mb4"
    )

    # 字段名要和 .env 里的 UPLOAD_PATH 对应（pydantic-settings 大小写不敏感）
    upload_path:Path = Path("data/uploads")
    max_upload_mb:int = 20

    deepseek_api_key:str=""
    deepseek_base_url:str="https://api.deepseek.com"
    deepseek_model:str="deepseek-v4-flash"

    #前端 dev server 的源，逗号分隔。声明成 str 而不是 list[str]：
    #pydantic-settings 对 list 字段会先尝试 JSON 解码，而 .env 里写的是裸 URL，会直接抛错。
    #默认同时给 localhost 和 127.0.0.1：Vite 起在 localhost，用户手输 127.0.0.1:5173 时
    #Origin 不匹配，会表现成一个很难查的 CORS 报错
    cors_origins:str="http://localhost:5173,http://127.0.0.1:5173"

    #关于向量数据库的配置
    #Milvus 是服务端（Docker standalone），不是本地文件库，所以配 URI 而不是路径
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "policy_child_chunks"
    embedding_dim: int = 512     # 必须和 embedding 模型一致（bge-small-zh-v1.5 = 512）

    #本地 embedding 模型（sentence-transformers / BGE）
    embedding_model: str = "bge-small-zh-v1.5"   # backend/models/ 下的子目录名，或 HuggingFace repo id
    embedding_device: str = "cpu"                # 有 CUDA 时可改 "cuda"
    embedding_batch_size: int = 32
    dashscope_api_key:str = ""

    #重排（cross-encoder）：召回之后重算相关性，并给出「证据是否足够」的判据
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # backend/models/ 子目录名，或 HuggingFace repo id
    enable_reranker: bool = True                     # 请求体里 rerank 省略时的默认值
    retrieval_top_k: int = 8                         # 开启重排时每个召回器取多少候选喂给重排模型
    rerank_top_k: int = 4                            # 重排后最多返回几条
    min_evidence_score: float = 0.25                 # 低于它就判为「证据不足」

    #RAG 问答（接口七）
    qa_temperature: float = 0.2                      # 政策问答要的是「贴着原文」，不是发挥
    qa_max_context_chars: int = 6000                 # 送进 LLM 的证据正文总字符上限

    model_config =SettingsConfigDict(
        # Anchor to backend/.env (this file is backend/app/config.py), regardless of cwd
        env_file= str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def resolved_upload_path(self) -> Path:
        """相对路径锚定到 backend/，这样从哪个目录启动服务结果都一样"""
        if self.upload_path.is_absolute():
            return self.upload_path
        return BACKEND_ROOT / self.upload_path

    @property
    def resolved_cors_origins(self) -> list[str]:
        """把逗号分隔的 CORS_ORIGINS 拆成列表，顺手丢掉空项和多余空白。

        容错「结尾多个逗号」「逗号后带空格」这类手写 .env 常见写法；空串会变成 []，
        那样浏览器一律不放行——好过悄悄放行一个不存在的源。
        """
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def resolved_embedding_model(self) -> str:
        """backend/models/<name> 存在就返回本地绝对路径；否则原样返回，交给 HF 按 repo id 处理"""
        candidate = BACKEND_ROOT / "models" / self.embedding_model
        return str(candidate) if candidate.is_dir() else self.embedding_model

    @property
    def resolved_reranker_model(self) -> str:
        """和 resolved_embedding_model 同一套规则。

        当前模型走 HF hub 缓存、不在 backend/models/ 下，但保留这条本地分支是为了
        将来零代码切换：把权重拷进 backend/models/ 再把 RERANKER_MODEL 改成目录名即可，
        不用记两套解析逻辑。
        """
        candidate = BACKEND_ROOT / "models" / self.reranker_model
        return str(candidate) if candidate.is_dir() else self.reranker_model

@lru_cache
def get_settings():
    return Settings()
