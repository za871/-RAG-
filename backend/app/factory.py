from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

from app.api import documents, qa

settings = get_settings()

def create_app() -> FastAPI:
    app=FastAPI(
        title=settings.app_name,
        description="FastAPI + LangChain 政策知识库问答系统",
        version="v1.0.0"
    )

    #CORS 必须在 include_router 之前挂：中间件的注册顺序决定了洋葱模型里谁在外层，
    #虽然对这个项目两者不冲突，但「中间件先配好再加路由」是更稳的习惯
    #allow_credentials 保持 False：前端不发 cookie，开着它反而要求 allow_origins 不能是通配
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.resolved_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #router 自己已经带了 /documents、/qa 前缀，这里只需要拼上统一的 /api。
    #rstrip("/") 不能省：prefix 以 / 结尾时 include_router 会直接抛 AssertionError
    app.include_router(documents.router, prefix=settings.api_prefix.rstrip("/"))
    app.include_router(qa.router, prefix=settings.api_prefix.rstrip("/"))

    @app.get("/")
    async def get_root():
        return\
            {
            "title":settings.app_name,
            "message": "欢迎来到政务RAG系统",
            "docs": "/docs",
        }

    return app

