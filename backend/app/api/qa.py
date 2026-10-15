import json
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.Schemas import AskRequest, AskResponse
from app.database import get_session
from app.services.qa_service import (
    EVENT_ERROR,
    LLMError,
    answer_question,
    prepare_answer,
    stream_answer,
)

#问答是另一个资源，不塞进 documents.py（那个文件的语义全是「文档 CRUD」）；
#独立 tag 也让 Swagger 里的分组清晰
router=APIRouter(prefix="/qa",tags=["qa"])

#RAG 问答：检索 → 没证据就拒答 → 组织上下文 → 约束提示词 → 调大模型 → 校验引用
#用同步 def 而不是 async def：里面是同步的检索 + 一次阻塞的 HTTP 调用，
#声明成普通函数 FastAPI 会把它放进线程池，不会卡住事件循环。
#（换成 async def + run_in_threadpool 会让 Session 跨线程使用，恰恰安全但很脆，不划算）
@router.post("/ask",response_model=AskResponse)
def ask(
        payload:AskRequest,
        session:Session = Depends(get_session)
):
    try:
        return answer_question(session,payload)
    except LLMError as e:
        #上游大模型不可用（网络/鉴权/限流/模型名不存在）：502，让调用方知道不是我们的逻辑错
        raise HTTPException(status_code=502,detail=str(e))
    except RuntimeError as e:
        #服务端自身配置问题：API key 没配、重排模型加载失败。503 比裸 500 更能说明性质
        raise HTTPException(status_code=503,detail=str(e))
    except ValueError as e:
        #过滤表达式里带了非法字符（引号/反斜杠）这类是用户输入问题，和 /search 保持一致的 400
        raise HTTPException(status_code=400,detail=str(e))


def _sse(event:str, payload:dict) -> str:
    """拼一帧 SSE。

    data 一律走 JSON 编码，而且不能省：答案正文里全是换行，直接塞进 `data:` 会让 SSE
    在第一个换行处就把帧截断。json.dumps 把 \\n 转义成字面的 \\n，一帧才稳。
    ensure_ascii=False 是为了中文在 curl 里肉眼可读，字节数也更小（SSE 本身是 UTF-8 字节流）。
    """
    return f"event: {event}\ndata: {json.dumps(payload,ensure_ascii=False)}\n\n"


#流式问答：先一次性把「检索 + 拒答判定 + 证据组装」跑完，再逐字把大模型的输出推给前端。
#事件顺序 meta → delta×N → done；拒答路径只发一帧 done，error 则可能在 delta 之间出现。
@router.post("/ask/stream")
def ask_stream(
        payload:AskRequest,
        session:Session = Depends(get_session)
):
    #prepare_answer 必须在 return 之前调用，绝不能挪进下面的生成器里：
    #生成器直到第一次 next() 才开始执行，那时响应头已经发出去了，HTTPException 再也改不了
    #状态码（会变成「200 里夹一个报错」）。放在这里，503「没配 key」和 400「过滤表达式非法」
    #就都保持了和 /ask 一样的语义
    try:
        prepared = prepare_answer(session, payload)
    except RuntimeError as e:
        raise HTTPException(status_code=503,detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e))

    def frames() -> Iterator[str]:
        """只做「语义事件 → SSE 帧」的翻译。真正的业务在 stream_answer 里。

        这里全程不碰 session：prepare_answer 已经把该查的库都查完了，
        而生成器的每次 next() 都跑在 Starlette 的线程池里，和建 Session 的线程不是同一个。
        """
        try:
            for event, body in stream_answer(prepared):
                yield _sse(event, body)
        except Exception as exc:
            #能走到这里，说明连 stream_answer 自己都没兜住（它已把大模型异常转成 error 事件）。
            #帧已经发出去一半、状态码改不了，唯一能做的是把原因也发出去——
            #否则前端只会看到连接断掉，无从判断发生了什么
            yield _sse(EVENT_ERROR, {"detail": f"服务端异常：{exc}"})

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        #Cache-Control 是给浏览器/代理的（SSE 不能被缓存）；
        #X-Accel-Buffering 是给 nginx 的——它默认会给响应加缓冲，那样流式就变成「攒够一次吐」
        headers={
            "Cache-Control":"no-cache",
            "X-Accel-Buffering":"no",
        },
    )
