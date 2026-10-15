from datetime import date

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.Schemas import (
    DeleteResponse,
    DocumentOut,
    DocumentPage,
    ResetResponse,
    SearchRequest,
    SearchResponse,
    TextIngestRequest,
)
from app.database import get_session
from app.models import DocumentStatus

from app.services.document_service import (
    delete_one,
    ingest_text,
    ingest_upload,
    list_documents,
    reset_all,
    search_documents,
)

#tags将来显示在swagger界面中让用户看到的
router=APIRouter(prefix="/documents",tags=["documents"])

#/api/documents  一定要是post请求才可以
#和 list_document/search 一样用普通 def：里面的 embedding 推理是 CPU 密集的
#（几千个 chunk 要十几秒），加上同步的 MySQL/Milvus 写入，写成 async def 会把这些
#全部压在事件循环上——实测 45KB 文本就能让整个服务假死 19 秒，并发请求全部排队。
#声明成 def，FastAPI 会把它放进线程池，事件循环不被占住
@router.post("",response_model=DocumentOut)
def upload_document(
        file:UploadFile = File(...),
        topic:str | None = Form(default=None),
        effective_date :date | None =Form(default=None),
        session:Session = Depends(get_session)
):
    #需要调用一个方法，真正实现file的上传，并返回一个DocumentOut对象
    try:
       #return await save_upload(session,file,topic,effective_date)
       return ingest_upload(session, file, topic, effective_date)
    except ValueError as e:
        #扩展名不支持、文件超限这类都是用户输入问题，属于400
        raise HTTPException(status_code=400,detail=str(e))

#查询所有文档，支持分页和按主题/状态筛选
#注意这里用 def 而不是 async def：里面是同步阻塞的数据库操作，
#声明成普通函数 FastAPI 会把它放到线程池里执行，不会卡住事件循环
@router.get("",response_model=DocumentPage)
def list_document(
        topic:str | None = Query(default=None,max_length=100),
        #标注成枚举，swagger会显示成下拉框，非法值由框架直接拦掉返回422
        status:DocumentStatus | None = Query(default=None),
        page:int = Query(default=1,ge=1),
        size:int = Query(default=20,ge=1,le=100),
        session:Session = Depends(get_session)
):
    items,total=list_documents(session,topic=topic,status=status,page=page,size=size)
    return DocumentPage(items=items,total=total,page=page,size=size)

#检索：给一段文本，返回最相关的子块（并带上父块正文）
#mode 决定走哪条路：vector=纯向量、bm25=纯关键词、hybrid=两路 RRF 融合（默认）
#同样是 def（同步的 embedding + Milvus + 数据库调用，交给线程池跑）
@router.post("/search",response_model=SearchResponse)
def search(
        payload:SearchRequest,
        session:Session = Depends(get_session)
):
    try:
        outcome=search_documents(
            session,
            payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
            topic=payload.topic,
            effective_before=payload.effective_before,
            #非法 mode 由 SearchRequest 的 Literal 在请求边界就拦成 422，这里不用再校验
            mode=payload.mode,
            rerank=payload.rerank,
        )
    except ValueError as e:
        #过滤表达式里带了非法字符（引号/反斜杠）这类是用户输入问题，属于 400
        raise HTTPException(status_code=400,detail=str(e))
    return SearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        mode=payload.mode,
        status=outcome.status,
        reason=outcome.reason,
        hits=outcome.hits,
    )

#纯文本入库：不传文件，直接把正文提交进来，走和上传一样的切片 + 向量化流程
@router.post("/text",response_model=DocumentOut,status_code=status.HTTP_201_CREATED)
def create_text_document(
        payload:TextIngestRequest,
        session:Session = Depends(get_session)
):
    try:
        return ingest_text(
            session,
            title=payload.title,
            content=payload.content,
            topic=payload.topic,
            effective_date=payload.effective_date,
        )
    except ValueError as e:
        #正文为空这类属于用户输入问题，和上传接口保持一致的 400
        raise HTTPException(status_code=400,detail=str(e))

#清空全部数据（重置）：放在 /{document_id} 之前更好读，两条路径本身不冲突
#用查询参数而不是请求体做确认：DELETE 带 body 会被不少代理/客户端直接丢掉
RESET_CONFIRM_TOKEN="RESET"
@router.delete("",response_model=ResetResponse)
def reset_documents(
        confirm:str = Query(...,description="必须传 confirm=RESET 才会清空全部数据"),
        session:Session = Depends(get_session)
):
    if confirm != RESET_CONFIRM_TOKEN:
        raise HTTPException(status_code=400,detail="清空全部数据需要 confirm=RESET")
    return ResetResponse(**reset_all(session))

#删除单个文档：MySQL 记录（连带切块）、Milvus 向量、磁盘文件一起删
@router.delete("/{document_id}",response_model=DeleteResponse)
def delete_document(
        document_id:str,
        session:Session = Depends(get_session)
):
    if not delete_one(session,document_id):
        raise HTTPException(status_code=404,detail="文档不存在")
    return DeleteResponse(deleted=True,document_id=document_id)
