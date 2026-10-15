import asyncio
import sys
from pathlib import Path

# 同 init_db.py：直接跑脚本时 sys.path[0] 是 scripts/，先把 backend/ 补进搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.model_factory import create_chat_model,invoke_text
from app.config import get_settings

async def main():
    s=get_settings()
    print(s)
    model= create_chat_model(settings=s,temperature=0.7)
    content,metadata=await invoke_text(model,"使用一句话，解释以下RAG是什么")
    print(content)
    print(metadata)

if __name__=='__main__':
    """s=get_settings()
    model = create_chat_model(settings=s,temperature=0.7)
    response=model.invoke("使用一句话，解释以下RAG是什么")
    #content和text是一样的
    print(response.content)
    print(response.text)
    #元数据中一般都是消耗的token信息
    print(response.usage_metadata)"""
    asyncio.run(main())