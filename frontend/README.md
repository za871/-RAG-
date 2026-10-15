# 政务政策 RAG 前端

单页问答界面：提问 → 答案逐字流式产出 → 引用卡片可点击溯源 → 拒答状态可辨识。

## 跑起来

后端必须先起好（见仓库根 README 的「快速开始」），前端才有东西可问：

```bash
# 1. 后端：Docker Desktop → Milvus → uvicorn（顺序见根 README）
cd backend && python -m uvicorn main:app --reload --port 8000

# 2. 前端
cd frontend
npm install
npm run dev          # http://localhost:5173
```

## 和 backend 的接口约定

| 前端调用 | 后端路由 |
| --- | --- |
| `askStream()` | `POST /api/qa/ask/stream`（SSE） |

用的是**流式**那条路，不是 `POST /api/qa/ask`——后者仍然存在且行为不变，可用来对照排查
（同一句话两边打一次，`answer` / `citations` / `status` 应当一致，因为两条路共用
`qa_service.finalize` 里的引用校验）。

### 后端地址

默认 `http://127.0.0.1:8000`。要改就在 `frontend/` 下建一个 `.env.local`：

```
VITE_API_BASE=http://127.0.0.1:8000
```

### CORS

前端直连后端（不配 Vite proxy），所以后端必须放行 5173：

- `backend/.env` 里的 `CORS_ORIGINS` 控制白名单，逗号分隔
- `backend/app/factory.py` 里的 `CORSMiddleware` 读它

改了 `CORS_ORIGINS` 要重启 uvicorn 才生效。**不要把 `vite.config.ts` 的
`strictPort` 关掉**——端口一旦漂到 5174，Origin 就和白名单对不上，
浏览器只会丢一个很含糊的 CORS 报错，很难查。

## 目录

```
src/
├─ types.ts              # 手抄自 backend/app/Schemas.py，改一边要改另一边
├─ api/errors.ts         # ApiError（区分「后端拒绝了」和「网络炸了」）
├─ api/sse.ts            # 字节流 → 事件的增量解码器，不碰 fetch，可单独测
├─ api/client.ts         # fetch + ReadableStream 调 /api/qa/ask/stream
├─ hooks/useAskStream.ts # 状态机 idle → retrieving → streaming → done / stopped / error
└─ components/
   ├─ QuestionForm.tsx   # 提问框 + 高级选项（mode / rerank / top_k / 过滤条件）
   ├─ AnswerView.tsx     # 正文，把 [n] 渲染成可点的引用角标
   ├─ CitationList.tsx   # 引用卡片，与角标互相高亮
   ├─ StatusNotice.tsx   # 三种拒答 + 引用校验失败的提示
   ├─ TimingBar.tsx      # 检索/生成/合计耗时、证据数、重排开关
   └─ DebugDetails.tsx   # 折叠的调试信息（发给 LLM 的原文、token 用量、引用校验）

scripts/
├─ test_sse.ts          # SSE 解析测试（node 24 原生跑 .ts，不需要 vitest）
└─ smoke_stream.py      # 不用浏览器也能看后端 SSE 事件的冒烟脚本
```

## 测试

```bash
npm run test:sse                              # 13 项合成用例，不需要后端
node scripts/test_sse.ts --live "人才引进有哪些补贴政策？"   # 再打一次真实后端，按 7 字节切块
```

`--live` 那一步是关键：一帧会被网络切成任意大小的 chunk，切点可能正好落在 `\n\n`、
`data:` 或一个中文字符的三个字节中间。固定按 7 字节切一遍，能把这类边界问题逼出来——
只在本地拼一整个响应体是测不到的。

```bash
# 后端 SSE 的原始事件流（不经过前端）
backend/.venv/Scripts/python.exe scripts/smoke_stream.py "问题"
```

## 两个容易踩的点

- **冷启动很慢不是 bug**：首次问答要加载重排权重（~2.2G）+ langchain-openai 首次初始化，
  10 秒以上是正常的。`TimingBar` 会把「检索」和「生成」拆开显示，便于判断慢在哪一段。
- **流式期间引用角标不可点**：`citations` 只在 `done` 事件里才有，而 `[n]` 从第一个
  delta 就出现在正文里了。此时角标降级成普通上标，`done` 到达后才变成可点。
