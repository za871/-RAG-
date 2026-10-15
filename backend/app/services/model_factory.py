
from typing import Iterator

from langchain.chat_models import init_chat_model


def create_chat_model(settings,temperature=0.5,stream_usage=False):
    s=settings
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    # model_provider 硬编码 "openai"（走 DeepSeek 的 OpenAI 兼容端点）是**唯一实测跑通**的组合，
    # 所以 .env 里的 MODEL_PROVIDER 刻意不接线：改成 "deepseek" 会走 langchain-deepseek
    # 另一条分支，参数名/鉴权/base_url 处理都不同，拿跑通的调用去换未验证的分支不划算
    #
    # stream_usage 必须显式传：ChatOpenAI 只在「用默认 base_url 且没传 client」时才自动打开它
    # （langchain_openai/chat_models/base.py:1171-1191），我们传了 DeepSeek 的 base_url，
    # 自动开启的条件不成立——不显式传，流式路径的 usage_metadata 就恒为空。
    # 对非流式无副作用：stream_options 只在 _stream/_astream 里才被拼进请求
    # （同文件 1585-1587、1844-1845），.invoke() 根本不会带上它
    model = init_chat_model(
        model=s.deepseek_model,
        model_provider="openai",
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        temperature=temperature,
        stream_usage=stream_usage,
    )
    return model


def invoke_messages(model, messages:list[dict]) -> tuple[str,dict]:
    """同步调用：把 system/user 消息列表交给模型，返回 (正文, usage_metadata)。

    用 .invoke() 而不是 .ainvoke()：调用方是 FastAPI 的同步 def handler，
    本身就在线程池里跑，阻塞的 HTTP 调用正是线程池要消化掉的东西；
    换成 async 反而会逼着检索那一侧也跨线程用 Session（见 README 接口七）。
    """
    response = model.invoke(messages)
    content = response.content
    # openai 兼容端点回的是 str；防御性归一化，免得个别 provider 回 content blocks 时炸掉
    if not isinstance(content, str):
        content = str(content)
    return content, dict(response.usage_metadata or {})


async def invoke_text(model,question):
    # ainvoke() 成功时返回的就是 AIMessage；失败时 API 会直接抛异常
    response=await model.ainvoke(question)
    return (response.content,dict(response.usage_metadata or {}))


def stream_messages(model, messages:list[dict], usage_sink:dict) -> Iterator[str]:
    """同步流式调用：逐个 yield 正文增量，流结束后把 usage_metadata 写进 usage_sink。

    用调用方传入的 dict 回传 usage，而不是当成生成器的返回值：usage 只在**流结束后**才拿得到，
    而生成器只有一个返回值、且那一刻已经在 StopIteration 里了。用 dict 也比「最后一个特殊元素」
    更好读——消费点不用挨个判断 `if isinstance(item, str)`。

    content 的归一化和 invoke_messages 保持一致：openai 兼容端点回的是 str，
    但个别 provider 会回 content blocks，那种情况下 str() 至少不炸。
    只 yield 非空增量：首帧常常是空串（role-only chunk），原样发出去只会让前端白更新一次。
    """
    for chunk in model.stream(messages):
        text = chunk.content
        if not isinstance(text, str):
            text = str(text)
        if text:
            yield text
        # 带 usage 的通常只有最后一帧（stream_options.include_usage 的效果）；
        # 用「后到覆盖先到」而不是只在第一帧取，免得某些实现把 usage 分几帧发
        meta = getattr(chunk, "usage_metadata", None)
        if meta:
            usage_sink.clear()
            usage_sink.update(dict(meta))
