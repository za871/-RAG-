# 政务政策 RAG 系统

基于 FastAPI + SQLAlchemy 2.0 的政务政策知识库后端，**RAG 全链路已跑通**。已实现**文档上传**、
**文档分页查询**、**纯文本入库**（`POST /api/documents/text`）、**混合检索 + 重排序**
（向量 + BM25 → RRF 融合 → cross-encoder 重排 → 证据阈值拒答，`POST /api/documents/search`）、
**RAG 问答**（检索 → 证据组装 → 约束提示词 → 大模型生成 → 引用校验，
`POST /api/qa/ask`）、**删除单个文档**（`DELETE /api/documents/{id}`）与
**清空全部数据**（`DELETE /api/documents?confirm=RESET`）。

## 技术栈

| 组件 | 版本 | 用途 |
| --- | --- | --- |
| FastAPI | 0.115.12 | Web 框架、自动生成 OpenAPI 文档 |
| uvicorn | 0.34.2 | ASGI 服务器 |
| SQLAlchemy | 2.0.41 | ORM（使用 `Mapped` / `mapped_column` 2.0 风格） |
| PyMySQL | 1.1.2 | MySQL 驱动 |
| MySQL | 8.x | 数据库 `gov_rag` |
| pydantic-settings | 2.9.1 | 从 `.env` 读取配置 |
| LangChain | 1.3.9 | LLM 调用（`model_factory.py` + `qa_service.py`，走 DeepSeek 的 OpenAI 兼容端点） |
| langchain-huggingface | 1.2.3 | 本地 embedding 模型加载 |
| sentence-transformers | 5.5.1 | 本地 embedding + cross-encoder 重排（`CrossEncoder`） |
| pymilvus | 3.0.0 | 向量库客户端（对应 Milvus server 3.0.x） |
| Milvus | 3.0.2 | 向量库（Docker standalone，见 `backend/deploy/`） |
| rank-bm25 | 0.2.2 | 内存 BM25 打分（混合检索的稀疏那一路） |
| jieba | 0.42.1 | BM25 的中文分词（精确模式） |
| pypdf / python-docx | 5.6.0 / 1.1.2 | PDF、DOCX 解析 |

完整依赖见 `backend/requirements.txt`。

## 目录结构

```
langchain_project/
├── README.md                     # 本文档
├── backend/
│   ├── main.py                   # 入口：app = create_app()
│   ├── requirements.txt
│   ├── .env                      # 实际配置（不要提交到版本库）
│   ├── .env.example              # 配置模板（当前内容已损坏，待重写）
│   ├── app/
│   │   ├── config.py             # Settings：读取 .env
│   │   ├── database.py           # engine / SessionLocal / get_session 依赖
│   │   ├── factory.py            # create_app()：注册路由
│   │   ├── models.py             # PolicyDocument / DocumentChunk ORM 模型
│   │   ├── Schemas.py            # Pydantic 出入参模型
│   │   ├── api/
│   │   │   ├── documents.py      # 文档相关路由（6 个接口）
│   │   │   └── qa.py             # RAG 问答路由（接口七）
│   │   └── services/
│   │       ├── document_service.py   # 文档业务逻辑：入库 / 检索 / 删除 / 重置
│   │       ├── chunking.py           # 父子切块（认标题 → 切章 → 大小两把尺子）
│   │       ├── milvus_store.py       # Milvus 读写封装、schema、过滤表达式
│   │       ├── bm25_store.py         # 内存 BM25 索引（语料来自 MySQL 子块）
│   │       ├── fusion.py             # RRF 融合（只用名次，不用分数）
│   │       ├── reranker.py           # cross-encoder 重排（sigmoid 归一化 + 证据阈值）
│   │       ├── prompt.py             # 提示词契约：约束关键词 / 证据渲染 / 引用解析
│   │       ├── qa_service.py         # RAG 问答流水线（检索 → 拒答 → 组装 → LLM → 校验）
│   │       ├── vector_store.py       # 本地 BGE embedding 模型
│   │       └── model_factory.py      # 大模型工厂（create_chat_model / invoke_messages）
│   ├── scripts/
│   │   ├── init_db.py            # 建表脚本
│   │   └── reindex_milvus.py     # 从 MySQL 重建 Milvus（schema 变更 / 半失败修复）
│   ├── deploy/                   # Milvus standalone 的 docker-compose
│   ├── models/                   # 本地 embedding 模型（bge-small-zh-v1.5）
│   ├── sample_data/              # 示例文档
│   ├── data/uploads/             # 上传文件落盘目录（运行时生成）
│   └── tests/sqlalchemy/         # SQLAlchemy 学习示例
└── frontend/                     # 前端（Vite + React + TS，问答界面）
    ├── src/api/sse.ts            # SSE 解析（不碰 fetch/import.meta，可脱离浏览器单独测）
    ├── src/api/client.ts         # fetch + ReadableStream 调 /api/qa/ask/stream
    ├── src/hooks/useAskStream.ts # 状态机 idle → retrieving → streaming → done/stopped/error
    ├── src/components/           # 提问框 / 答案 / 引用卡片 / 状态提示 / 计时条 / 调试信息
    └── scripts/test_sse.ts       # SSE 解析测试（node 24 原生跑 .ts，不需要 vitest）
```

## 快速开始

```bash
cd backend

# 1. 创建并激活虚拟环境
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备配置：复制模板并填写数据库连接与 API Key
cp .env.example .env

# 4. 建表（幂等，已存在的表不会被改动）
python scripts/init_db.py

# 5. 启动 Milvus（向量库，接口三/四依赖它）
#    需要先启动 Docker Desktop，然后：
cd deploy && docker compose up -d && cd ..
curl http://localhost:9091/healthz        # 期望 OK，首次启动要 30~60 秒
# 注意：镜像必须在本地已有，或能从 docker.1ms.run 拉到；
#       docker compose 并发 pull 多个镜像容易卡死，必要时逐个 pull

# 6. 启动服务
python -m uvicorn main:app --reload --port 8000
```

启动后：

- Swagger UI：<http://127.0.0.1:8000/docs>
- OpenAPI JSON：<http://127.0.0.1:8000/openapi.json>
- 根路径：<http://127.0.0.1:8000/>

路由表（`app.include_router` 会把 `settings.api_prefix` 拼在 router 自带的 `/documents` 前缀前面）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/documents` | 上传文档（[接口一](#接口一上传文档)） |
| GET | `/api/documents` | 分页查询文档列表（[接口二](#接口二分页查询文档列表)） |
| POST | `/api/documents/text` | 纯文本入库（[接口三](#接口三纯文本入库post-apidocumentstext)） |
| POST | `/api/documents/search` | 混合检索 + 重排序 + 证据阈值（[接口四](#接口四混合检索--重排序post-apidocumentssearch)） |
| DELETE | `/api/documents/{document_id}` | 删除单个文档（[接口五](#接口五删除文档delete-apidocumentsdocument_id)） |
| DELETE | `/api/documents?confirm=RESET` | 清空全部数据（[接口六](#接口六清空全部数据delete-apidocumentsconfirmreset)） |
| POST | `/api/qa/ask` | RAG 问答（[接口七](#接口七rag-问答post-apiqaask)） |
| POST | `/api/qa/ask/stream` | 同上，但以 SSE 流式返回（[接口八](#接口八流式问答post-apiqaaskstream)） |

---

## 接口一：上传文档

`POST /api/documents`，`multipart/form-data`。

### 请求参数

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 是 | 文档文件，仅允许 `.pdf` / `.docx` / `.txt` |
| `topic` | string | 否 | 主题分类，最长 100 字符，用于检索时缩小范围 |
| `effective_date` | date | 否 | 政策生效日期，格式 `YYYY-MM-DD` |

### 处理流程

1. **校验扩展名**：取文件名后缀并转小写，不在白名单内则抛 `ValueError`
2. **校验大小**：读取全部内容，超过 `MAX_UPLOAD_MB`（默认 20MB）则抛 `ValueError`
3. **落库（先占位）**：创建 `PolicyDocument` 记录，`stored_path` 暂时留空
4. **`flush()` 取出主键**：`id` 由 Python 侧 `uuid4()` 生成，`flush` 后即可读取
5. **写文件**：以 **`<uuid><后缀>`** 命名写入上传目录
6. **`commit()` + `refresh()`**：提交事务，并回读数据库生成的 `created_at` / `updated_at`

### 两个设计要点

**为什么用 uuid 命名而不是原文件名？**
用户上传的文件名不可信，可能包含 `../` 等路径穿越字符，也可能重名互相覆盖。用 uuid 命名后，
原始文件名只作为 `file_name` 字段存进数据库，磁盘上永远是安全的 uuid。这也是为什么第 4 步要先
`flush` 拿到 `id` 才能写文件。

**为什么用 `model_validate(document)` 构造返回值？**
如果手工拼 `DocumentOut(...)` 的每个字段，很容易和数据库里真实存的值不一致（比如状态、时间戳写错）。
`refresh()` 之后直接用 `DocumentOut.model_validate(document)`，由 ORM 对象反向映射出响应，
落库值和返回值天然一致。

### 状态机

`status` 字段有三个取值（对应数据库的 `enum('pending','ready','failed')`）：

```
pending ──┬──> ready    切片 + 向量化成功，可被检索
          └──> failed   解析失败，原因写入 error_message
```

当前**上传后一律是 `pending`**，因为切片和向量化流水线尚未实现。等索引流程接入后，
后台任务处理成功再改写为 `ready`。在流水线落地前把状态标成 `ready` 是不诚实的——
前端会以为文档已经可检索。

### 示例

```bash
curl -X POST "http://127.0.0.1:8000/api/documents" \
  -F "file=@sample_data/policy.txt" \
  -F "topic=人才引进" \
  -F "effective_date=2024-05-01"
```

```json
{
  "id": "8f88d7db-9efc-40a9-ab2d-f1e690241b24",
  "filename": "policy.txt",
  "content_type": "text/plain",
  "topic": "人才引进",
  "effective_date": "2024-05-01",
  "status": "pending",
  "error_message": null,
  "chunk_count": 0,
  "index_version": 0,
  "created_at": "2026-09-28T16:56:12"
}
```

不支持的扩展名返回 **400**（不是 500）：

```json
{ "detail": "仅支持PDF、DOCX、TXT文件" }
```

---

## 接口二：分页查询文档列表

`GET /api/documents`

### 请求参数

| 参数 | 类型 | 默认 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `topic` | string | 无 | 最长 100 | 按主题**精确匹配**筛选 |
| `status` | enum | 无 | `pending` / `ready` / `failed` | 按状态筛选 |
| `page` | int | 1 | ≥ 1 | 页码，从 1 开始 |
| `size` | int | 20 | 1 ~ 100 | 每页条数 |

`topic` 和 `status` 都不传时返回全部文档；同时传则按 **AND** 叠加。

### 响应结构

```json
{
  "items": [
    {
      "id": "8f88d7db-9efc-40a9-ab2d-f1e690241b24",
      "filename": "人才政策.txt",
      "content_type": "text/plain",
      "topic": "人才引进",
      "effective_date": "2024-05-01",
      "status": "pending",
      "error_message": null,
      "chunk_count": 0,
      "index_version": 0,
      "created_at": "2026-09-28T16:56:12"
    }
  ],
  "total": 1,
  "page": 1,
  "size": 20
}
```

- `items` —— 当前页的数据（`DocumentOut` 列表）
- `total` —— **符合筛选条件的总条数**，与 `page` / `size` 无关，前端据此算总页数：`ceil(total / size)`
- `page` / `size` —— 回显本次请求的分页参数，便于前端确认

### 实现原理

#### 1. 分页用 `LIMIT` / `OFFSET`

```python
.offset((page - 1) * size).limit(size)
```

第 1 页偏移 `0`，第 2 页偏移 `size`，以此类推。`page` 必须从 1 开始，
否则 `(page-1)*size` 会算出负数偏移量。

#### 2. `total` 与分页数据复用同一组筛选条件

这是本接口最容易写错的地方。`total` 必须用**独立的 `COUNT` 语句**取，且
**筛选条件要和分页查询完全一致**，否则前端算出的总页数会和实际数据对不上。

代码里的做法是先把条件构造进一个列表，然后两处共用：

```python
conditions = []
if topic:
    conditions.append(PolicyDocument.topic == topic)
if status is not None:
    conditions.append(PolicyDocument.status == status)

# 总数：独立的 count 语句
total = session.scalar(
    select(func.count()).select_from(PolicyDocument).where(*conditions)
) or 0

# 当前页：同样的 conditions + 分页
statement = (
    select(PolicyDocument)
    .where(*conditions)
    .order_by(PolicyDocument.created_at.desc(), PolicyDocument.id.desc())
    .offset((page - 1) * size)
    .limit(size)
)
```

`SELECT count(*)` 是**独立的 SQL 语句**，不会被分页语句的 `LIMIT` 影响，
所以它统计的永远是全量条数。

#### 3. 排序为什么要带 `id` 作为第二排序键

```python
.order_by(PolicyDocument.created_at.desc(), PolicyDocument.id.desc())
```

MySQL 的 `datetime` 类型**默认只有秒级精度**，没有小数位。如果同一秒内上传了多条文档，
它们的 `created_at` 完全相同，此时只按 `created_at` 排序，数据库返回这些行的相对顺序是
**未定义的**（取决于执行计划和物理存储顺序）。

后果是 `LIMIT/OFFSET` 翻页时会出现**重复或遗漏**：比如第 1 页最后一条和第 2 页第一条
可能是同一条记录，而另一条记录从两页之间"消失"。这是 OFFSET 分页的经典陷阱。

加上唯一的 `id` 作为第二排序键，排序就变成**全序**（total order）——任意两条记录都能
比出先后，翻页结果就稳定了。实测三条文档在同一秒入库，用 `?size=1` 逐页翻三页，
拿到的是三个互不重复的 id。

#### 4. 为什么 `topic` 用精确匹配

`policy_documents` 表上有 `ix_policy_documents_topic` 索引。`PolicyDocument.topic == topic`
生成的 `WHERE topic = 'xxx'` 能命中该索引；如果改成 `contains()`（即 `LIKE '%xxx%'`），
**前置通配符会让 B+ 树索引失效**，退化成全表扫描。将来的模糊搜索应该交给全文索引或向量检索，
而不是在这个接口上牺牲索引。

`status` 同理，走 `ix_policy_documents_status` 索引。

#### 5. 为什么路由函数用 `def` 而不是 `async def`

```python
@router.get("", response_model=DocumentPage)
def list_document(...):      # 注意：不是 async def
```

SQLAlchemy 的 `Session` 是**同步阻塞**的，一次查询会占用工作线程等待数据库返回。
如果把路由声明成 `async def`，这段阻塞代码会跑在事件循环里，**并发请求会全部排队**，
吞吐量退化成单线程。

声明成普通 `def` 时，FastAPI 会自动把它**丢进线程池**执行，多个请求可以并行处理。
（对比：`POST` 上传接口是 `async def`，因为 `await upload.read()` 是真异步的。）

### 错误码约定

| 情况 | 状态码 | 说明 |
| --- | --- | --- |
| `page=0` / `page=-1` / `size=0` / `size=101` | **422** | 超出 `Query(ge=, le=)` 约束 |
| `status=bogus` | **422** | 非法枚举值，响应里会列出允许的取值 |
| `topic` 超过 100 字符 | **422** | 列宽约束，避免超长参数打到数据库 |
| `page` 超出最后一页 | **200** | 返回 `items: []`，这是正常结果而非错误 |
| 筛选无匹配 | **200** | `total: 0`，`items: []` |

`status` 参数直接标注成 `DocumentStatus` 枚举类型，因此：

1. Swagger UI 中会自动渲染成**下拉框**，不会让用户手输
2. 非法值由 FastAPI 在**进入路由函数之前**拦掉，返回 422，业务代码里不必再写校验

深翻页（`page=9999`）返回 200 + 空列表是**有意为之**：REST 语义里"没有更多数据"不是错误，
前端翻到最后一页后应该看到空列表而不是报错。

### 示例

```bash
# 默认：第 1 页，每页 20 条
curl "http://127.0.0.1:8000/api/documents"

# 按主题筛选
curl "http://127.0.0.1:8000/api/documents?topic=人才引进"

# 按状态筛选
curl "http://127.0.0.1:8000/api/documents?status=pending"

# 组合筛选 + 自定义分页
curl "http://127.0.0.1:8000/api/documents?topic=人才引进&status=pending&page=1&size=10"
```

数据库为空时：

```json
{ "items": [], "total": 0, "page": 1, "size": 20 }
```

> **Windows 控制台提示**：在 GBK 代码页的终端（如 cmd、部分 Git Bash 配置）里用 `curl` 传中文
> 参数时，shell 会按 GBK 编码发送字节，而服务端按 UTF-8 解码，结果是中文变成乱码存入数据库
> （例如 `人才` 存成 `ÈË²Å`）。这是终端编码问题，**不是接口 bug**。
>
> 规避方式任选其一：
> - 用 Python 客户端发送，如 `httpx.get(URL, params={"topic": "人才引进"})`
> - 使用 URL 编码后的参数：`curl "http://127.0.0.1:8000/api/documents?topic=%E4%BA%BA%E6%89%8D%E5%BC%95%E8%BF%9B"`
> - 在终端先执行 `chcp 65001` 切到 UTF-8 代码页
> - 直接在 Swagger UI（`/docs`）里操作，浏览器始终使用 UTF-8

---

## 接口三：纯文本入库（`POST /api/documents/text`）

不传文件，直接把一段正文提交进来，走和上传文件**完全相同的**切片 + 向量化流程。
适合把一小段政策、FAQ 问答对直接塞进知识库，不必先落一个文件。

### 请求参数（JSON body）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `content` | str | **是** | 正文，至少 1 个字符；全空白（如 `"   "`）会被拒 |
| `title` | str | 否 | 存进 `file_name` 列，最长 225 字符；不传默认 `text-input` |
| `topic` | str | 否 | 主题，检索时可用来过滤 |
| `effective_date` | date | 否 | 生效日期，检索时可用来过滤 |

### 设计要点

**两条入库路径共用一个核心函数**。`ingest_upload()` 和 `ingest_text()` 都调用
`document_service._chunk_persist_and_index(session, document, pages)`：
切块 → 落 MySQL（父块 `flush` 后回填子块的 `parent_id`）→ `commit` → embedding →
写 Milvus → 置 `ready`。两条路径的差别只在最前面：一个是从文件抽 `(页码, 文本)`，
另一个是把整篇正文当成一页（页码为 `None`，和 docx/txt 的口径一致）。

```
POST /documents        ──extract_pages()──┐
                                          ├──> _chunk_persist_and_index()
POST /documents/text   ──[(None, content)]┘        │
                                                   └─> 切片 → MySQL → Milvus
```

这样做的收益是**没有第二套流水线要维护**：切片策略、Milvus 字段、状态机、
失败处理都只有一份实现，改一处两边同时生效。

**两个列必须给值**（都是 NOT NULL）：

- `content_type` 固定写 `"text/plain"`
- `stored_path` 写空串 `""` —— 没有磁盘文件。删除接口要**靠它为空来跳过删文件**，
  注意 `Path("")` 会塌缩成当前目录，所以那里的判断必须是 `path.is_file()` 而不是 `exists()`

### 错误码约定

| 情况 | 状态码 | 说明 |
| --- | --- | --- |
| `content` 为空字符串 | **422** | Pydantic `min_length=1` 拦掉 |
| `content` 只有空白字符 | **400** | 通过校验但业务层拒绝（`ValueError` → 400） |
| `title` 超过 225 字符 | **422** | 列宽约束 |
| 请求体多传了字段 | **422** | `APIModel` 设了 `extra="forbid"`，不静默忽略 |

### 示例

```bash
curl -X POST "http://127.0.0.1:8000/api/documents/text" \
  -H "Content-Type: application/json" \
  -d '{
        "title": "人才引进政策",
        "topic": "人才",
        "effective_date": "2023-01-01",
        "content": "第一条 人才引进\n本市对高层次人才给予安家补贴。\n第二条 落户条件\n硕士及以上学历可直接落户。"
      }'
```

返回 `201`，`status` 直接是 `ready`（同步做完切片和向量化）：

```json
{
  "id": "92ff1719-99bd-4a5c-aa1a-d3f851d0a456",
  "filename": "人才引进政策",
  "content_type": "text/plain",
  "topic": "人才",
  "effective_date": "2023-01-01",
  "status": "ready",
  "error_message": null,
  "chunk_count": 2,
  "index_version": 1,
  "created_at": "2026-10-12T21:21:44"
}
```

---

## 接口四：混合检索 + 重排序（`POST /api/documents/search`）

给一段自然语言问题，返回知识库里最相关的**子块**，并附带每个子块所属**父块的正文**。
证据不足时**拒答**（见下文 `status`）。

检索走**两路**，用 `mode` 选择；召回之后再由 `rerank` 决定要不要过 cross-encoder 重排：

| `mode` | 走哪条路 | 说明 |
| --- | --- | --- |
| `vector` | 只走向量 | `rerank:false` 时与加 BM25 之前**完全一致**；开启重排后会多取候选并重排，见下方说明 |
| `bm25` | 只走 BM25 | 纯关键词，不调 embedding；对精确术语/数字/条款号最稳 |
| `hybrid` | 两路 + RRF 融合（**默认**） | 各取候选后按名次融合，见下文 |

> **重排是叠加在 `mode` 之上的一个阶段，不是第四种 mode。** 三种 mode 都可以开或关重排。
>
> ⚠️ **默认行为**：`.env` 里 `ENABLE_RERANKER=true`，而请求体的 `rerank` 默认跟随它，
> 所以**默认就是开着重排的**。这意味着 `top_k=8` 的请求默认最多只回 `RERANK_TOP_K=4` 条，
> 排序依据也从 `distance`/`rrf_score` 变成 `rerank_score`。想要「重排前」的原始行为，
> 请显式传 `rerank: false`。

### 请求参数（JSON body）

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | str | **是** | — | 问题文本，至少 1 个字符 |
| `top_k` | int | 否 | `8` | 返回条数，范围 1~50 |
| `document_id` | str | 否 | — | **只在这一篇文档里检索** |
| `topic` | str | 否 | — | **只检索这个主题**（最长 100 字符） |
| `effective_before` | date | 否 | — | **只检索生效日期在这一天（含）之前的政策** |
| `mode` | str | 否 | `hybrid` | 检索模式，见上表；非法值直接 **422** |
| `rerank` | bool | 否 | 跟随 `ENABLE_RERANKER`（`true`） | 是否对本批召回做 cross-encoder 重排；传 `false` 可单次关闭 |

三个过滤条件可以任意组合，条件之间是 **AND**；**三条都对两路同时生效**（见下文「两路的过滤口径必须一致」）。

### 处理流程

```
用户提问
   │
   ├─ mode = vector ────────────→ 1a. embed_query(query)          ← 自动带 BGE 检索指令前缀
   │                                 Milvus 向量检索，带过滤表达式
   │
   ├─ mode = bm25 ──────────────→ 1b. jieba 分词
   │                                 内存 BM25 打分，过滤在内存里做
   │
   ├─ mode = hybrid（默认） ─────→ 1a + 1b 都跑
   │                                 └─ 2b. RRF 融合（只用名次）
   │
   ├─ 3. rerank 开启时：对候选池（pool_k 条）逐对跑 cross-encoder
   │      └─ sigmoid 归一化到 0~1 → 丢掉 < MIN_EVIDENCE_SCORE → 取 ≤ min(top_k, RERANK_TOP_K)
   │         └─ 一条都不剩 → 拒答（status=refused，hits 为空，不再往下走）
   │
   ├─ 4. 对最终命中的子块取 parent_id
   │
   └─ 5. 一次 select … where id in (parent_ids) 回 MySQL 取父块正文
```

### 为什么要 cross-encoder 重排

RRF 融合的是两路**召回向**的列表，**只用名次、从不重算相关性**：一个在 BM25 里名次高、
但语义其实不搭的块，RRF 照样把它排前面。cross-encoder 把 `(query, 正文)` 拼成一个序列
**一起**过模型，直接给「这一对有多相关」打分，比双塔余弦 / 词频都准。

代价是它慢得多（每对都要一次完整前向），所以**只对少量候选跑**，不能拿来做召回。

### 重排分怎么算：逐对 sigmoid

```
rerank_score = sigmoid(交叉编码器原始 logit) = 1 / (1 + exp(-logit))
```

为什么是逐对 sigmoid 而不是 softmax / min-max：

- sigmoid 是**逐对**算的，分数只取决于这一对 `(query, chunk)`，**与候选集合无关**
  → 同一个 chunk 在不同 `top_k`、不同候选池下分数不变，**跨查询可比**，阈值才立得住
- softmax / min-max 依赖候选集合，候选一换分数就变，`MIN_EVIDENCE_SCORE=0.25` 这种固定阈值
  就失去意义了

代价是分数不再「加和为 1」——本来也不需要。

> **`MIN_EVIDENCE_SCORE=0.25` 配 v2-m3 实际偏严**。实测同一批候选里，
> 「硕士落户需要什么条件」对「硕士及以上学历可直接落户」= **0.91**，
> 对相关但不回答问题的「本市对高层次人才给予安家补贴」只有 **0.008**，
> 对完全无关的句子 ≈ **1.6e-5**。v2-m3 打分非常自信、区分度极大，
> 所以 0.25 这条线已经能把「沾边但不答问题」的块筛掉了。但它是**启发式起点、
> 没有离线评测支撑**（和 `RRF_K=60` 一样），拿到真实语料后应该按需调。

### 候选池为什么缩到 `RETRIEVAL_TOP_K`

重排要对 `(query, 全文)` 逐对跑一个 568M 参数的模型，CPU 上约 **0.1~0.2 秒/对**。
上游 `fusion_candidate_k(top_k)` 最多会给 32 条，直接喂进去就是 3~6 秒的额外延迟。所以：

```
final_k = min(top_k, RERANK_TOP_K)                          # 重排后最终条数
pool_k  = min(max(RETRIEVAL_TOP_K, final_k), fusion_candidate_k(top_k))
```

默认值下 `pool_k = 8`（不是 32）。`max(..., final_k)` 保证 `top_k` 比 `RETRIEVAL_TOP_K`
大时候选池够用；外层的 `fusion_candidate_k` 保证今天的天花板不被突破。
**重排开启时四种 mode 的候选池一律是 `pool_k`，延迟与 mode 无关。**

### 最终条数规则

| rerank | mode | 每个召回器的 limit | 重排输入 | 最终条数 |
| --- | --- | --- | --- | --- |
| 关 | `vector` / `bm25` | `top_k` | — | `top_k` |
| 关 | `hybrid` | `fusion_candidate_k(top_k)` | — | RRF 融合后 `top_k` |
| 开 | 任意 | `pool_k` | 前 `pool_k` 条 | 阈值过滤后 ≤ `min(top_k, RERANK_TOP_K)` |

**阈值先过滤、再按 `top_k` 截断**，顺序不能反：反过来 `top_k` 会把「本来过线但排在后面」
的证据挡掉，阈值就失去了「判证据够不够」的意义。

### 为什么要有 BM25 这一路

纯稠密向量擅长语义泛化（"落户要什么学历"能命中"硕士及以上可直接落户"），
但对**精确的字面量**很不敏感：条款编号、金额（`10000 元`）、专有名词、
生僻术语，很容易被语义相似度糊过去。BM25 恰好相反——它只看词是否出现及其稀有度，
所以两路是互补的：**一路管"意思像"，一路管"字面上有"**。

`rank-bm25==0.2.2` 本来就写在 `requirements.txt` 里但没被使用，这次把它接上。

### RRF 为什么不用加权求和

向量的 `distance` 是余弦相似度（0~1 区间），BM25 的分数是无上界的实数，
**两者量纲完全不同**，直接加权求和得先调归一化、再调权重，换个语料就得重调，不可迁移。

RRF（Reciprocal Rank Fusion）**只用名次、不用分数**：

```
score(d) = Σ_r  1 / (k + rank_r(d))          rank 从 1 开始，k = 60
```

- `k=60` 是 Cormack et al. 2009 的经验值（代码里的 `fusion.RRF_K`），不是调优结果
- 两路各自取 `candidate_k = clamp(top_k × 4, 20, 100)` 条候选再融合（`fusion_candidate_k`）。
  多取一些是为了给「另一路也命中」的块留出加分空间
- 一个块被两路同时命中 → 两边的 `1/(k+rank)` 相加 → 自然排在只有一路命中的前面
- **并列的打破规则**：`dict` 保插入序 + `sorted` 是稳定排序，所以同分时「先出现的排前面」，
  也就是向量路的次序起主导。同样输入必然同样输出，RRF 可复现

### 两路的过滤口径必须一致

`document_id` / `topic` / `effective_before` 三个条件对两路都生效，而且**语义必须逐条对齐**，
否则同一个查询换了 `mode` 就会返回不同的命中集合。为此：

- 日期那条两路都用 `milvus_store.effective_date_to_int` 转换成 `YYYYMMDD` 的整数再比大小。
  未填生效日期的文档存的是 `DATE_MAX`，会被 `<= X` 自然排除——两路的行为天然一致
- BM25 语料只收 `status == ready` 的文档。向量路只能检索到成功写进 Milvus 的块
  （`failed` 文档的向量已被清掉、`indexing` 的还没写），不加这个过滤 BM25 就会多召回

### BM25 索引为什么在内存里

BM25 是**语料级**统计（IDF 依赖整个语料），所以必须手里有一份全量语料：

- 语料来自 **MySQL 的 `document_chunks`**（`is_parent=False` 的子块），
  装载时 JOIN `policy_documents` 把 `topic` / `effective_date` 冗余进内存——
  因为 chunk 表本身没有这两列，但过滤又需要它们
- 进程内缓存，**入库 / 删除 / 重置时显式失效**（见 `bm25_store.invalidate_bm25_index`）
- 查询时是**一个全局索引 + 按谓词掩码**，不是每个过滤组合重建索引。
  重建会让「同一句话在不同过滤下 IDF 不同、分数不一致」，反而更难解释；
  全局 IDF 在过滤子集上打分是标准做法，也便宜得多
- **不需要改 Milvus schema、不需要 reindex**：BM25 索引完全独立于向量库

> **多 worker 注意**：缓存是进程内的，`uvicorn --workers N` 下 worker A 的写入
> 不会失效 worker B 的缓存。当前是单进程部署，够用。
> 另外 `scripts/reindex_milvus.py` 是**独立进程**，它可能把 `failed` 文档修成 `ready`
> 从而改变 BM25 语料——**reindex 之后请重启 API**。

### 为什么过滤要下推到 Milvus，而不是检索完再筛

这是本接口**最容易写错的地方**。看起来"先检索 top_k 再按主题/日期扔掉不符合的"更简单，
但那样会**丢召回**：

```
问：补贴政策是什么？
Milvus 返回 top_k=8 条 → 其中只有 1 条属于 topic=人才 → 就只能返回 1 条

可是库里明明还有 3 条人才相关的政策，只是向量相似度排在第 9、12、15 位，被 top_k 截断了
```

要事后过滤就得把 `top_k` 放大很多倍（放多大都不保险），而**下推给 Milvus 是精确的**：
Milvus 在候选集上先做标量过滤再算相似度，返回的 `top_k` 条**全部**满足条件。

### `topic` / `effective_date` 为什么在 Milvus 里也存了一份

这两列原本**只存在于 MySQL 的 `policy_documents`**，Milvus 的 collection 里没有，
所以在 Milvus 侧根本无从过滤。解决方案是**冗余**一份进 Milvus 的 schema
（见 [Milvus 集合结构](#milvus-集合结构)），写入时从文档上带过去。
两个系统各自保存同一份数据的取舍：

- **收益**：过滤能下推，精确且不必回表，检索路径只碰 Milvus
- **代价**：写代码时要记得两边一起改；改主题/日期需要重新向量化（目前**没有**改文档的接口，
  要改就删掉重新入库）

### 两个哨兵值

Milvus 的**标量字段不支持 null**，所以"没填"的字段要拿一个哨兵值顶替，读取时再还原：

| 字段 | 哨兵 | 还原规则 |
| --- | --- | --- |
| `topic` | 空串 `""` | `""` → `None` |
| `effective_date` | `99991231` | `== DATE_MAX` → `None` |
| `page` | `-1`（`NO_PAGE`） | `< 0` → `None` |
| `parent_id` | 空串 `""` | `""` → `None` |

`DATE_MAX` 取 `99991231` 而不是 `0` 是**刻意的**：这样 `effective_date <= X`
会把"没填生效日期"的文档**排除**掉。生效日期未知时无从判断它是否在 `X` 之前生效，
宁可漏也不要错引一份政策。想让未填日期的文档也被 `<= X` 捞进来，把哨兵改成 `0` 即可。

### `distance` / `bm25_score` / `rrf_score` / `rerank_score` 的语义

四个分数字段都是**越高越相关**，但含义和"能不能当排序依据"完全不同：

| 字段 | 含义 | 何时有值 | 备注 |
| --- | --- | --- | --- |
| `distance` | 余弦相似度（0~1） | 该块被**向量路**命中时 | 字段名有误导性，是 Milvus 为了 API 统一才叫 distance；换 `L2` 度量才是"越小越好" |
| `bm25_score` | BM25 原始分（无上界） | 该块被 **BM25 路**命中时 | 只在同一语料内可比 |
| `rrf_score` | RRF 融合分 | 只在 `mode=hybrid` | **只在同一次响应内可比**，跨请求没有意义 |
| `rerank_score` | `sigmoid(交叉编码器 logit)`，落在 (0,1) | 只在**重排开启**时 | **逐对计算 ⇒ 跨查询可比**，这是它和上面三个最大的区别 |

**排序依据取决于重排开没开，客户端不要重排**：

- **重排开启（默认）**：返回顺序**统一是 `rerank_score` 降序**，三种 `mode` 都一样。
  此时 `distance` / `bm25_score` / `rrf_score` 仍然填着，但它们**只是这条块原本
  怎么被召回的痕迹，不是排序依据**——重排不会把原始分数抹掉，就是为了让你能看出
  「这条是靠哪一路进来的、RRF 给它排第几」
- **重排关闭（`rerank: false`）**：
  - `mode="vector"`：顺序就是余弦相似度降序，`distance` 是排序依据
  - `mode="bm25"`：顺序就是 BM25 降序，`distance` 恒为 `null`
  - `mode="hybrid"`：顺序是 **RRF 顺序**，`distance` / `bm25_score` 都只是各自那一路的
    原始分，不是排序依据；一个块可能只有向量分（`bm25_score: null`）或只有 BM25 分（`distance: null`）
  - 此时 `rerank_score` 恒为 `null`

想要「重排前」的原始排序，就显式传 `rerank: false`，不要自己去重排这些分数字段。

### `status` / `reason`：拒答

`status` **恒定存在**（不随重排开关而改变字段有无，否则同一个查询在两个部署上会返回
不同形状的响应），三态：

| `status` | 含义 | `hits` | `reason` |
| --- | --- | --- | --- |
| `ok` | 有证据 | 非空 | `null` |
| `no_evidence` | **召回本身就是空**：库为空、过滤条件全不中、`query` 分词后为空 | `[]` | `"检索未命中任何证据"` |
| `refused` | **召回到了，但全部低于 `MIN_EVIDENCE_SCORE`** | `[]` | `"召回证据与问题相关性不足，已拒答"` |

两者都返回空的 `hits`，但含义不同、上层处理方式也不同：`no_evidence` 是「库里真的没有」，
`refused` 是「有相关材料但不够格回答」——所以才会给 `status` 分三态，而不是只给一个
`answerable: bool`。

> **`refused` 只在重排开启时可能发生**（重排关了就没有「相关性分数」可判）。
> 重排关闭时的空召回同样是 `no_evidence`——`no_evidence` 是关于**召回**的事实，与重排无关。

### 错误码约定

| 情况 | 状态码 | 说明 |
| --- | --- | --- |
| `query` 为空 | **422** | `min_length=1` |
| `top_k` 超出 1~50 | **422** | `ge=1, le=50` |
| `topic` 超过 100 字符 | **422** | 列宽约束 |
| `mode` 不是三个合法值之一 | **422** | `Literal` 在请求边界就拦下 |
| `rerank` 不是能转成 bool 的值 | **422** | 注意 pydantic 宽松模式会把 `"yes"`/`"true"`/`"1"`/`0` 强转成 bool |
| `document_id` / `topic` 含引号或反斜杠 | **400** | 会被拼进 Milvus 过滤表达式，属于注入，直接拒 |
| `query` 分词后为空（如全是标点） | **200** | `status: "no_evidence"`、`hits: []`——BM25 无 token 可打，不是错误 |
| 库为空 / 重置后检索 | **200** | `status: "no_evidence"`、`hits: []`，**不是错误** |
| 条件互斥（无匹配） | **200** | `status: "no_evidence"`、`hits: []` |
| 重排后证据全部低于阈值 | **200** | `status: "refused"`、`hits: []`，**不是错误** |
| 重排模型本地没有权重 | **500** | 服务端配置问题（不是用户输入），提示在服务端日志里 |

> **为什么含引号就拒**：Milvus 的过滤表达式**没有参数绑定**，而 pymilvus 客户端
> 对表达式**不做任何校验**、原样转发给服务端。所以 `"` 和 `\` 必须自己挡掉，
> 否则可以闭合字符串往表达式里塞条件。见 `document_service._filter_literal`。
>
> 注意这是**有意的分叉**：`mode="bm25"` 时过滤是纯 Python 字符串比较、没有注入面，
> 所以 `topic` 含引号**不会被拒**；只有 `mode` 为 `vector` / `hybrid` 时才会 400。

### 示例

```bash
# 1) 全库检索
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "硕士落户需要什么条件", "top_k": 3}'

# 2) 只在一篇文档里检索
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "落户条件", "document_id": "92ff1719-99bd-4a5c-aa1a-d3f851d0a456"}'

# 3) 只看人才主题
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "补贴", "topic": "人才"}'

# 4) 只看 2024-01-01 之前生效的政策
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "补贴", "effective_before": "2024-01-01"}'

# 5) 三个条件叠加（AND）
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "补贴", "topic": "人才", "effective_before": "2024-01-01",
       "document_id": "92ff1719-99bd-4a5c-aa1a-d3f851d0a456"}'

# 6) 只要关键词命中，不要语义泛化
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "10000 元", "mode": "bm25"}'

# 7) 显式拿稠密排序（等价于加 BM25 之前的行为）
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "硕士落户需要什么条件", "mode": "vector"}'

# 8) 关掉重排，拿「重排前」的原始顺序（可以单次关闭做对比）
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "硕士落户需要什么条件", "rerank": false}'

# 9) 问一个库里没有的问题 → 200 + status:"refused" + hits:[]
curl -X POST "http://127.0.0.1:8000/api/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "公司年假怎么算"}'
```

响应（`parent_content` 就是 small-to-big 回查回来的父块正文，可直接喂给 LLM 当上下文。
默认 `hybrid` + 默认重排开启，第一条两路都命中、第二条只在 BM25 里命中）：

```json
{
  "query": "硕士落户需要什么条件",
  "top_k": 3,
  "mode": "hybrid",
  "status": "ok",
  "reason": null,
  "hits": [
    {
      "chunk_id": "5bae646f-822a-4bbf-af39-67bb50b07d9e",
      "document_id": "92ff1719-99bd-4a5c-aa1a-d3f851d0a456",
      "parent_id": "ecfb1b7f-076c-41bb-97c1-b14cfac14165",
      "content": "第二条 落户条件\n硕士及以上学历可直接落户。",
      "section": "第二条 落户条件",
      "page": null,
      "ordinal": 1,
      "distance": 0.4829743504524231,
      "bm25_score": 6.8312,
      "rrf_score": 0.032522,
      "rerank_score": 0.9111,
      "topic": "人才",
      "effective_date": "2023-01-01",
      "parent_content": "第一条 人才引进\n本市对高层次人才给予安家补贴。\n第二条 落户条件\n硕士及以上学历可直接落户。"
    },
    {
      "chunk_id": "c81d2f47-1a0e-4f6b-8c31-2f8b0a9d4e77",
      "document_id": "92ff1719-99bd-4a5c-aa1a-d3f851d0a456",
      "parent_id": null,
      "content": "第三条 补贴标准\n硕士一次性安家补贴 10000 元。",
      "section": "第三条 补贴标准",
      "page": 2,
      "ordinal": 4,
      "distance": null,
      "bm25_score": 4.1177,
      "rrf_score": 0.016129,
      "rerank_score": 0.0259,
      "topic": "人才",
      "effective_date": "2023-01-01",
      "parent_content": null
    }
  ]
}
```

> 第二条的 `distance` 是 `null`——它只在 BM25 那一路里命中，没有向量相似度。
> 这就是为什么 `distance` 从必填改成了可选。
>
> 两条的 `distance` / `bm25_score` / `rrf_score` 都还在，但**排序依据是 `rerank_score`**：
> 第一条 0.91 远高于第二条 0.026。原始分数保留下来是为了让你能看出「这条是靠哪一路进来的」。

拒答响应（召回有结果，但全部低于 `MIN_EVIDENCE_SCORE=0.25`）：

```json
{
  "query": "公司年假怎么算",
  "top_k": 8,
  "mode": "hybrid",
  "status": "refused",
  "reason": "召回证据与问题相关性不足，已拒答",
  "hits": []
}
```

库里根本没有相关材料时则是 `status: "no_evidence"` + `reason: "检索未命中任何证据"`。

**为什么返回的是子块命中、父块正文**：子块（420 字）用来做向量检索以命中精确语义，
父块（1200 字）用来给 LLM 提供足够上下文。详见 `milvus_store.py` 顶部的说明。

---

## 接口五：删除文档（`DELETE /api/documents/{document_id}`）

把一篇文档**从三个地方一起清掉**：Milvus 向量、MySQL 记录（连带切块）、磁盘上的上传文件。

### 删除顺序（重要）

顺序和入库时**正好相反**，这是刻意设计的：

```
1. Milvus 删向量          ← 先删。失败则整个请求失败，MySQL 行还在，重试即可（删除是幂等的）
2. MySQL 删子块 → 删父块 → 删文档
3. 磁盘文件（失败不阻断）
```

入库时的顺序是「先 MySQL 后 Milvus」（见接口一的注释）。删除取反，是为了让**最坏情况
永远是可重试的**：

| 顺序 | 中途失败后的状态 | 能否恢复 |
| --- | --- | --- |
| **Milvus → MySQL**（本实现） | 向量没了、记录还在 | ✅ 重试一次即可，`delete_document` 幂等 |
| MySQL → Milvus | 记录没了、向量成孤儿 | ❌ API 再也够不着那些向量，只能手工清理 |

### 为什么先删子块再删父块

`document_chunks.parent_id` 是**自引用外键**。MySQL 逐行处理删除时，如果父行先于子行被删，
会撞上外键约束报 **1451**。先子后父则无论外键是 `CASCADE` 还是 `RESTRICT` 都安全。

另外这里用的是 SQLAlchemy 的**批量 `delete()`**（而不是 ORM 的 `session.delete()`），
它**不触发** `relationship` 上的 `cascade="all,delete-orphan"`，所以顺序是硬要求；
每句都带 `.execution_options(synchronize_session=False)`，避免它去核对马上要丢弃的对象。

### 一个实测到的坑：Milvus 删除后必须 flush

`client.delete()` 只是写了一条**墓碑记录**，不 `flush` 的话随后的 `search`
**仍会把已删的子块检索出来**——删完立刻再问一句是很正常的用法，不能指望恰好有别的写入
帮忙 flush。已在 `milvus_store.delete_document` 里补上 `client.flush()`
（和 `upsert_child_chunks` 里那句同一个道理）。

### 示例

```bash
curl -X DELETE "http://127.0.0.1:8000/api/documents/92ff1719-99bd-4a5c-aa1a-d3f851d0a456"
# -> {"deleted": true, "document_id": "92ff1719-99bd-4a5c-aa1a-d3f851d0a456"}

# 再删一次（幂等）：404 文档不存在
```

---

## 接口六：清空全部数据（`DELETE /api/documents?confirm=RESET`）

清空**全部**知识库：MySQL 两张表、Milvus 整个 collection、上传目录里的文件。
**表结构保留**（不像 drop table 那样连表一起删）。

### 为什么确认口令要用查询参数而不是请求体

DELETE 请求带 body 会被不少代理 / HTTP 客户端**直接丢掉**（RFC 也没规定 DELETE 的 body 语义），
参数就静默失效了。放在 query string 上最稳。

```bash
# 不带 confirm -> 422（Query(...) 是必填）
# confirm 值不对 -> 400
curl -X DELETE "http://127.0.0.1:8000/api/documents?confirm=RESET"
# -> {"documents_deleted": 4, "chunks_deleted": 12, "files_deleted": 2}
```

### 为什么是 drop collection 而不是按条件 delete

Milvus 的按条件 `delete` 只写墓碑，**segment 不会回收、HNSW 索引也不会释放**，
语义上不算"清空"。`drop_collection` 才是真的清干净。

代价是"集合没了"，但这**不需要任何补救代码**：`upsert_child_chunks` 每次入库都会调
`ensure_collection`，集合不存在就按 schema 重新建好（含索引）并 load。
重置之后直接重新入库即可，验证过是通的。

### 重置之后想恢复数据：`scripts/reindex_milvus.py`

`document_chunks` 表里**父块子块的正文都在**，MySQL 是元数据的真相来源，
所以**不依赖原始上传文件**就能把整个 Milvus 重建出来：

```bash
cd backend && python scripts/reindex_milvus.py
```

```text
已删除旧 collection，开始从 MySQL 重建……
  人才引进政策: 2 条
  税务政策: 2 条
重建完成：2 个文档，4 条子块
```

这个脚本同时是 **Milvus schema 变更时的迁移路径**，也是"MySQL 有块、Milvus 没有"
那种半失败状态的修复手段（跑完会把 `status` 修正回 `ready`）。

---

## 接口七：RAG 问答（`POST /api/qa/ask`）

给一个自然语言问题，走完整条 RAG 链路，返回**回答 + 引用 + 耗时 + 调试信息**。

### 请求参数（JSON body）

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | str | **是** | — | 问题文本，至少 1 个字符 |
| `top_k` / `document_id` / `topic` / `effective_before` / `mode` / `rerank` | — | 否 | 同接口四 | **与 `/search` 逐字段对齐**——问答的召回范围必须和检索完全同构，否则「同一句话在 `/search` 能召回、在 `/qa/ask` 不能」会变成排查负担 |

**没有 `history` / `session_id`**：这个接口是一轮一问，没有会话概念。

### 处理流程

```
用户问题
   │
   ├─ 1. 调用检索服务（就是接口四那条链路，含重排与阈值）
   │
   ├─ 2. status != ok ？
   │      └─ 是 → 直接拒答，**不构造 prompt、不建 model client**，返回罐装文案
   │
   ├─ 3. 组织上下文：按 parent_id 折叠命中 → 取父块正文 → 编号成【证据 1..N】
   │
   ├─ 4. 构造约束关键词（拼进 system prompt，见下）
   │
   ├─ 5. 调用大模型生成回答
   │
   ├─ 6. 校验引用：抽出所有 [n]，越界的从正文剔除并记入 debug
   │
   └─ 7. 返回 {回答, 引用, 耗时, 调试信息}
```

### 四态 status

| `status` | 含义 | 由谁判定 | `answer` | `citations` |
| --- | --- | --- | --- | --- |
| `ok` | 正常作答 | — | 模型输出 | 模型真正引用到的证据 |
| `no_evidence` | 库里没有相关材料 | **检索侧** | 罐装拒答文案 | `[]` |
| `refused` | 有材料但相关性不足（全低于 `MIN_EVIDENCE_SCORE`） | **检索侧** | 罐装拒答文案 | `[]` |
| `insufficient_evidence` | 材料相关，但不足以回答这个问题 | **大模型侧** | 罐装拒答文案 | `[]` |

前两个是从 `/search` 原样透传的；第三个是模型「看过证据后说自己答不了」。
三种"答不出来"对上层是三种不同事实，所以给了三态而不是一个 `answerable: bool`。

> **拒答时一定不调大模型**：`status != ok` 直接返回，省一次调用，也彻底断了「让模型硬答」的可能。
> 这在验证里是用「`create_chat_model` 一被调用就抛异常」的哨兵断言过的。

### 约束关键词（system prompt）

这是防幻觉最主要的闸。拼进 system prompt 的约束是：

```
你是政务政策知识库的问答助手。约束关键词：只依据证据作答、逐句标注引用、禁止编造、证据不足就拒答。
```

后面跟 5 条硬规则：① 只能用【证据】里的内容，不得用外部知识或推测补充；
② 每条事实陈述后必须紧跟 `[1]` / `[1][2]` 形式的证据编号，编号只能是 1..N；
③ 严禁编造编号、条款、数字、日期、文件名；④ 证据不足以回答时**只输出** `INSUFFICIENT_EVIDENCE`；
⑤ 中文、先结论后分点、不复述提示词。完整原文见 `app/services/prompt.py`。

### 引用校验：`[n]` 越界剔除

从答案里正则抽出所有 `[n]`，号码不在 `1..N` 内的一律**从正文里删掉**、
记进 `debug.invalid_citation_indices`、并把 `citations_valid` 置为 `false`。

三个必须用正则（而不是 `text.replace("[n]","")`）才能处理的坑，都有测试覆盖：

| 输入 | 期望 | 为什么 |
| --- | --- | --- |
| `[10]`，N=2 | 整体删掉，**不留 `0`** | `replace("[1]")` 会把 `[10]` 误伤成 `0` |
| `[2]0`，N=2 | 原样保留、尾随的 `0` 不动 | 只匹配到 `[2]`，不该吃掉后面的字符 |
| `[1][2]` | 两个标记各自独立判定 | 两个不重叠匹配 |

`citations_valid` 的判定：`status != ok` 时恒为 `true`（压根没有答案断言，也就没有伪造引用）；
`status == ok` 时要求**「没有越界编号」且「至少有一条引用」**——一个非拒答的答案却一条引用都没有，
等于整段无据，这正是要暴露的异常，不能因为"没有越界编号"就判它干净。

> ⚠️ **已知的假阳性**：政策原文里的年号方括号（如 `[2024]`）会被当成越界引用删掉。
> 提示词已要求模型不要复述标题，但这条靠 `≥N 即越界` 兜住，属零成本误伤。

### 证据为什么按 `parent_id` 折叠

同一父块的多个子块可能同时被召回（hybrid 下很常见），而 `parent_content` **完全相同**。
不折叠就会把同一段父块正文重复喂给模型、还会产生重复引用。
折叠后：一条证据对应一个父块，`chunk_id` 取该父块下排名最高的那个子块作代表，
`index` 从 1 连续编号——**编号就是引用号**。

上下文超 `QA_MAX_CONTEXT_CHARS` 时**从尾部整条丢弃证据**（不截断正文）：
截断会破坏 `【证据 n】` 的完整性、把证据切一半送进去比少给一条更糟。
丢完 `index` 仍然连续 `1..N`，引用校验天然正确。第一条无条件保留，
避免出现「检索说 ok 却没有证据」时提示词里的「1 到 0」没法看。

### 为什么这个接口是同步 `def`

检索是同步 SQLAlchemy/Milvus，大模型调用用的是**同步** `model.invoke()`。
两者都是阻塞操作，声明成普通 `def` 让 FastAPI 丢进线程池即可，不需要 async。

换成 `async def` + `run_in_threadpool` 会把 `Depends(get_session)` 创建的 Session
拿到另一条线程上使用——**恰好安全但很脆**（SQLAlchemy 的 Session 非线程安全，
当前靠"只用一次、串行"维持，谁加一个 `asyncio.gather` 就静默坏掉），收益远小于隐患。

### 耗时字段

| 字段 | 含义 |
| --- | --- |
| `retrieval_ms` | 检索总耗时。**重排并在这里，不单列**——重排在 `search_documents` 内部，拆出来得改已冻结的 `SearchOutcome` |
| `llm_ms` | 大模型调用耗时。拒答路径恒为 `0.0` |
| `total_ms` | 端到端，含证据组装、filename 回查、引用解析，所以一定 ≥ 前两者之和 |

> **冷启动**：`total_ms` 里包含每进程一次的初始化成本。实测首次 `create_chat_model` 约 **3.7 秒**
> （langchain-openai 的 import + 客户端初始化），之后每次调用是 **0ms**；叠加首次加载重排模型的
> **6.3 秒**，一个冷进程的第一次问答大约要 10 秒以上。预热钩子（`WARMUP_ON_STARTUP`）仍未接线，
> 运维上靠部署后先打一次 `/api/qa/ask`。

### 调试信息

`debug` **恒定返回**（用户明确要求能拿到拼好的完整 prompt），含：

| 字段 | 说明 |
| --- | --- |
| `model` / `temperature` | 实际用的模型与温度 |
| `rerank` | 实际生效的重排开关（`None` 已在服务层解析成具体 bool，不会含糊） |
| `retrieval_status` / `retrieval_reason` | 检索侧的状态与原因 |
| `hit_count` / `evidence_count` / `evidence_dropped` / `evidence_chars` | 命中数（折叠前）/ 证据数（折叠后）/ 预算丢弃数 / 证据总字数 |
| `evidence_scores` | 每条证据的代表块与它的 `rerank_score` |
| `llm_called` | 是否真的调了模型（拒答路径为 `false`） |
| `messages` | ★ **送进模型的 system + user 完整原文**，逐字回放 |
| `usage` | 模型回的 token 用量 |
| `cited_indices` / `invalid_citation_indices` | 引用到的编号 / 越界的编号 |
| `llm_insufficient` | 模型是否回了不足标记 |
| `answer_raw` | 模型原始输出（角标剔除前），排查看「模型到底说了什么」 |

`messages` 直接复用送进 `model.invoke()` 的那个 list，不是重新拼一遍——
避免出现「debug 里显示的 prompt 和真正发出去的不一致」这种最没用的调试信息。

### 错误码

| 情况 | 状态码 | 说明 |
| --- | --- | --- |
| `query` 为空 / `mode` 非法 | **422** | Pydantic 在请求边界拦下 |
| `document_id` / `topic` 含引号或反斜杠 | **400** | 会被拼进 Milvus 过滤表达式，与 `/search` 一致 |
| 大模型调用失败（网络/鉴权/限流/模型名不存在） | **502** | 上游不可用，不是我们的逻辑错 |
| 服务端配置问题（`DEEPSEEK_API_KEY` 没配、重排模型加载失败） | **503** | 比裸 500 更能说明性质 |
| 库里没材料 / 证据不足 | **200** | `status` 区分 `no_evidence` / `refused` / `insufficient_evidence`，**不是错误** |

> 注意 `/search` 遇到 `RuntimeError` 仍是 500，而 `/qa/ask` 映射成 503——语义上 503 更准，
> 但两个接口不一致，属于有意为之。

### 示例

```bash
curl -X POST "http://127.0.0.1:8000/api/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "硕士落户需要什么条件？补贴多少？"}'
```

响应（截断）：

```json
{
  "query": "硕士落户需要什么条件？补贴多少？",
  "status": "ok",
  "reason": null,
  "answer": "结论：硕士落户需满足学历和社保两项条件，可获一次性安家补贴 10000 元。\n\n1. 落户条件：具有硕士及以上学历，且在本市连续缴纳社会保险满 1 年[1]。\n2. 补贴标准：一次性发放安家补贴 10000 元[2]。",
  "citations": [
    { "index": 1, "chunk_id": "r1", "document_id": "d1", "filename": "人才政策.pdf",
      "section": "第二条 落户条件", "page": 1, "topic": "人才",
      "effective_date": "2023-01-01", "snippet": "第一条 人才引进\n本市对高层次人才给予安家补贴。…",
      "rerank_score": 0.9 },
    { "index": 2, "chunk_id": "r2", "document_id": "d2", "filename": "补贴办法.pdf",
      "section": "第三条 补贴标准", "page": 2, "topic": "人才",
      "effective_date": "2023-01-01", "snippet": "第三条 补贴标准\n对具有硕士学历的人才，一次性发放安家补贴 10000 元。",
      "rerank_score": 0.5 }
  ],
  "citations_valid": true,
  "timings": { "retrieval_ms": 10.34, "llm_ms": 1954.81, "total_ms": 5492.39 },
  "debug": {
    "model": "deepseek-v4-flash",
    "temperature": 0.2,
    "rerank": true,
    "retrieval_status": "ok",
    "hit_count": 2,
    "evidence_count": 2,
    "evidence_dropped": 0,
    "llm_called": true,
    "usage": { "input_tokens": 407, "output_tokens": 237, "total_tokens": 644 },
    "cited_indices": [1, 2],
    "invalid_citation_indices": [],
    "messages": [
      { "role": "system", "content": "你是政务政策知识库的问答助手。约束关键词：…" },
      { "role": "human", "content": "【证据】\n【证据 1】\n来源文件：人才政策.pdf\n…" }
    ]
  }
}
```

拒答响应：

```json
{
  "query": "公司年假怎么算",
  "status": "no_evidence",
  "reason": "检索未命中任何证据",
  "answer": "知识库中没有检索到与该问题相关的政策依据，因此无法作答。请补充问题细节或先上传相关政策文档。",
  "citations": [],
  "citations_valid": true,
  "timings": { "retrieval_ms": 12.4, "llm_ms": 0.0, "total_ms": 12.9 },
  "debug": { "llm_called": false, "messages": [], "usage": {} }
}
```

---

## 接口八：流式问答（`POST /api/qa/ask/stream`）

入参与 [接口七](#接口七rag-问答post-apiqaask) **完全一致**（`AskRequest`），响应改为
`text/event-stream`，让答案逐字抵达前端，而不是等整段生成完。

### 事件序列

```
event: meta     data: {"status":"ok","retrieval_ms":3475.1,"hit_count":3,"evidence_count":2,
                       "evidence_dropped":0,"rerank":true,"model":"deepseek-v4-flash"}
event: delta    data: {"text":"河南"}          ← 多次，就是逐字增量
event: done     data: <与接口七同构的完整 AskResponse>
```

- `meta` 只发一次，在检索（含重排）**跑完之后**、大模型开始生成之前。检索冷启动要加载
  重排权重（~2.2G，实测 6 秒+），这段时间界面靠它把「检索中」换成「生成中」。
- `done` 一定存在且是最后一帧。**拒答路径只发 `done`、没有 `meta`/`delta`**，
  其 `data` 就是接口七那条早退分支产出的同一个 `AskResponse`——前端两条路径共用一套渲染。
- `error` 可能在 `delta` 之间出现，见下面「状态码」。

### 状态码

| 阶段 | 失败时 | 为什么 |
| --- | --- | --- |
| 检索（`prepare_answer`） | 正常状态码：400 / 503 | 它在 `return StreamingResponse(...)` **之前**执行，所以还能改状态码 |
| 大模型生成 | 200 + `event: error` | 响应头早发出去了，状态码改不了 |

这是刻意的取舍：把 `prepare_answer` 放在路由里而不是生成器里，正是为了保住 400/503 的语义
（生成器直到第一次 `next()` 才执行，那时再抛 `HTTPException` 只会变成「200 里夹一个报错」）。
代价就是大模型阶段只能走 `error` 事件——前端据此出错误条和重试按钮。

### 为什么生成器不碰 Session

`StreamingResponse` 会在 Starlette 的线程池里逐个 `next()` 这个同步生成器，而 SQLAlchemy
`Session` 是处理请求的那个线程建的。所以所有 DB 访问（检索、filename 回查）都收在
`prepare_answer` 里，生成器只消费纯数据——`qa.py` 里关于「Session 跨线程」的顾虑
（见接口七的注释）在这条路上从设计上就不存在。

### 与非流式的关系

两条路共用 `qa_service.prepare_answer`（检索 + 拒答判定 + 证据 + 提示词）和
`qa_service.finalize`（引用解析 + 校验 + 组装），差别只在中间那一步用 `.invoke()` 还是
`.stream()`。所以同一句话问两次，两边的 `status` / `citations` 语义必然一致，
不会出现「流式说 ok、非流式说 refused」这种最费解的偏差。

> `create_chat_model(..., stream_usage=True)` 是流式路径拿得到 `usage_metadata` 的前提：
> `ChatOpenAI` 只在用默认 base_url 时才自动开 `stream_usage`，而这里传的是 DeepSeek 的
> `base_url`，不显式传的话 `debug.usage` 恒为 `{}`。它对非流式没有副作用——
> `stream_options` 只在 `_stream`/`_astream` 里才被拼进请求。

### 示例

```bash
curl -N -X POST http://127.0.0.1:8000/api/qa/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"人才引进有哪些补贴政策？"}'
```

前后端的联调脚本（按 7 字节切块打这个接口，验证帧切分）：
`node frontend/scripts/test_sse.ts --live "问题"`。

---

## 数据库表结构

`policy_documents`：

```sql
CREATE TABLE `policy_documents` (
  `id` varchar(36) NOT NULL,
  `file_name` varchar(225) NOT NULL,
  `stored_path` varchar(500) NOT NULL,
  `content_type` varchar(100) NOT NULL,
  `topic` varchar(100) DEFAULT NULL,
  `effective_data` date DEFAULT NULL,
  `status` enum('pending','ready','failed') NOT NULL,
  `error_message` text,
  `chunk_count` int NOT NULL,
  `index_version` int NOT NULL,
  `created_at` datetime NOT NULL DEFAULT (now()),
  `updated_at` datetime NOT NULL DEFAULT (now()),
  PRIMARY KEY (`id`),
  KEY `ix_policy_documents_status` (`status`),
  KEY `ix_policy_documents_topic` (`topic`),
  KEY `ix_policy_documents_effective_data` (`effective_data`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
```

### ⚠️ 关于 `file_name` 和 `effective_data` 两列

数据库里这两列的名字是**历史遗留**，和第 4 列 `effective_data` 一样，`effective_data`
是早期写代码时的**拼写错误**（应为 `effective_date`）。

**Python 侧的属性名已经修正，数据库列名保持不变**：

```python
filename: Mapped[str] = mapped_column("file_name", String(225))
effective_date: Mapped[date | None] = mapped_column("effective_data", Date, index=True)
```

`mapped_column()` 的**第一个位置参数就是数据库里真实的列名**，Python 属性名取自类属性名。
这样做的收益：

- **不需要数据库迁移**。`Base.metadata.create_all()` 对已存在的表是 no-op（只 `CREATE TABLE IF NOT EXISTS`），
  如果直接改模型属性名而不指定列名，模型和真实表会**永久不一致**，之后所有针对该列的查询都会报
  `Unknown column`。
- 索引名自动生成为 `ix_policy_documents_effective_data`，**与表中已有索引完全一致**。
- API 响应里是 `filename` / `effective_date`（见 `DocumentOut`），**对前端的契约是干净的**，
  不需要 `validation_alias` 之类的桥接。

**改动这一处时务必注意**：不要"顺手"把 `mapped_column("file_name", ...)` 里的字符串删掉，
也不要动数据库列名。项目目前**没有接入 Alembic 等迁移工具**，改列名只能手工 `ALTER TABLE`
并连带 `RENAME INDEX`，风险远大于收益。

## Milvus 集合结构

collection 名来自配置 `MILVUS_COLLECTION`（默认 `policy_child_chunks`）。
**只往里写子块**，父块留在 MySQL 供回查（small-to-big）。

> 混合检索**不往这里加任何字段、也不需要 reindex**：BM25 索引是另一套完全独立的结构，
> 语料直接来自 MySQL 的 `document_chunks`（见 `bm25_store.py`）。改动只在应用侧。
>
> 重排序同理：**不需要任何新 Milvus 字段、不需要重建 collection**。
> cross-encoder 在内存里对已召回的 chunk 重打分，压根不碰向量库。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_id` | VARCHAR(36) | **主键**，就是 MySQL `document_chunks.id`，两边靠它对齐 |
| `vector` | FLOAT_VECTOR(512) | bge-small-zh-v1.5 的向量，已做 L2 归一化 |
| `document_id` | VARCHAR(36) | 所属文档，删除 / 限定文档检索用 |
| `parent_id` | VARCHAR(36) | 父块 id，回 MySQL 取上下文用；无父块时为空串 |
| `content` | VARCHAR(8192) | 子块正文（冗余一份，省一次回表） |
| `section` | VARCHAR(512) | 章节标题 |
| `page` | INT64 | 页码，docx/txt 为 `-1`（`NO_PAGE`） |
| `ordinal` | INT64 | 文档内序号 |
| `topic` | VARCHAR(512) | **主题，从 MySQL 冗余**，检索过滤用 |
| `effective_date` | INT64 | **生效日期 `YYYYMMDD`，从 MySQL 冗余**，检索过滤用；未填为 `99991231` |

索引：

| 字段 | 索引类型 | 用途 |
| --- | --- | --- |
| `vector` | `HNSW`（`metric_type=COSINE`） | 向量检索 |
| `topic` | `INVERTED` | `topic == "…"` 等值过滤 |
| `document_id` | `INVERTED` | `document_id == "…"` 等值过滤、删除定位 |
| `effective_date` | `STL_SORT` | `effective_date <= X` 范围过滤 |

### ⚠️ 改 schema 必须重建集合

**Milvus 不支持加列**。往 `_build_schema()` 里加了字段之后，**已存在的老 collection
不会自动升级**，它的字段列表就少了一列。这时检索会用到一个不存在的字段名，
Milvus 只会回一句很难懂的"未知字段"。

所以 `ensure_collection()` 里加了一道**显式校验**：集合已存在时用
`describe_collection()` 取字段名集合，缺字段就抛 `RuntimeError`，错误信息里直接点名
让你跑 `scripts/reindex_milvus.py`。这是刻意的选择——**检索路径上静默少召回，比直接报错危险得多**。

> **VARCHAR 的 `max_length` 是按 UTF-8 字节算的，不是字符数**。中文 1 个字 3 字节，
> 所以 `topic` 取的是 512 而不是照抄 MySQL 那列的 100（100 字符最多 300 字节）。
> 另外 pymilvus **客户端不校验长度**，超了要等插入时才由服务端报错。
> `content` 给 8192 也是同理（子块最长 420 字 ≈ 1260 字节）。

---

## 配置说明（`backend/.env`）

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `APP_NAME` | str | `""` | 应用名称，显示在 Swagger 标题 |
| `DEBUG` | bool | `true` | 调试开关 |
| `API_PREFIX` | str | `/api` | 所有接口的统一前缀。**不能带尾斜杠**，否则 `include_router` 会抛 `AssertionError` |
| `DATABASE_URL` | str | 本机 MySQL | SQLAlchemy 连接串，`mysql+pymysql://用户:密码@主机:端口/库名?charset=utf8mb4`，`charset=utf8mb4` 必须保留以支持中文 |
| `UPLOAD_PATH` | Path | `data/uploads` | 上传文件落盘目录。填相对路径时锚定到 `backend/`（见 `Settings.resolved_upload_path`） |
| `MAX_UPLOAD_MB` | int | `20` | 单文件大小上限（MB） |
| `DEEPSEEK_API_KEY` | str | `""` | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | str | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | str | `deepseek-v4-flash` | 模型名 |
| `MILVUS_URI` | str | `http://localhost:19530` | Milvus 是服务端（Docker standalone），所以配 URI 而不是本地路径 |
| `MILVUS_TOKEN` | str | `""` | 本地 standalone 不带鉴权；**传空串**，传了反而可能被当成非法凭证 |
| `MILVUS_COLLECTION` | str | `policy_child_chunks` | collection 名，改这个等于换一个向量库 |
| `EMBEDDING_MODEL` | str | `bge-small-zh-v1.5` | `backend/models/` 下的子目录名，或 HuggingFace repo id |
| `EMBEDDING_DEVICE` | str | `cpu` | 有 CUDA 时可改 `cuda` |
| `EMBEDDING_BATCH_SIZE` | int | `32` | embedding 批大小 |
| `EMBEDDING_DIM` | int | `512` | **必须和 embedding 模型一致**（bge-small-zh-v1.5 = 512），建集合时用 |
| `RERANKER_MODEL` | str | `BAAI/bge-reranker-v2-m3` | 重排模型；`backend/models/` 子目录名或 HuggingFace repo id。**别改回 `bge-reranker-base`**——它本地没有权重（HF 缓存里只是个空 refs 指针），加载要联网下 ~1.1G |
| `ENABLE_RERANKER` | bool | `true` | 请求体里 `rerank` 省略时的默认值。**注意这意味着默认就开着重排** |
| `RETRIEVAL_TOP_K` | int | `8` | 开启重排时每个召回器取多少候选喂给重排模型（见「候选池为什么缩到 RETRIEVAL_TOP_K」） |
| `RERANK_TOP_K` | int | `4` | 重排后最多返回几条。所以 `top_k=8` 默认只回 4 条 |
| `MIN_EVIDENCE_SCORE` | float | `0.25` | 归一化后的证据阈值，低于它判为「证据不足」；全被滤掉就拒答 |
| `QA_TEMPERATURE` | float | `0.2` | 问答的采样温度。比默认 0.5 低——政策问答要的是「贴着原文」，不是发挥 |
| `QA_MAX_CONTEXT_CHARS` | int | `6000` | 送进 LLM 的证据正文总字符上限；超了从尾部整条丢证据（不截断，保证 `[n]` 有效） |

> 配置类的字段名对应 `.env` 的键名（pydantic-settings 大小写不敏感，但不做同义词映射——
> 字段叫 `upload_path`，`.env` 里写 `UPLOAD_DIR` 是**读不到**的，会静默落回默认值）。

---

## 已知问题 / TODO

- **`.env` 中有真实 API Key**：项目当前**没有 `.gitignore`**。若这个目录会进版本库，
  请先轮换 `DEEPSEEK_API_KEY` 并加上 `.gitignore`（至少忽略 `.env`、`.venv/`、`data/`、`__pycache__/`）。
- **`.env.example` 内容已损坏**：存在大量拼写错误、缺失下划线、键值粘连（如
  `EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5RERANKER`）、以及一个真实 Key。需要按 `.env` 重写，
  且模板中不应包含任何真实密钥。
- **重排序与问答相关配置已接线**：`RERANKER_MODEL`、`ENABLE_RERANKER`、`RETRIEVAL_TOP_K`、
  `RERANK_TOP_K`、`MIN_EVIDENCE_SCORE`、`QA_TEMPERATURE`、`QA_MAX_CONTEXT_CHARS` 七个键
  都有对应的 `Settings` 字段并且真的被读取了。
  **仍未接线**的是：`ENABLE_QUERY_REWRITE`、`ENABLE_LLM_JUDGE`、`WARMUP_ON_STARTUP`、
  `WARMUP_LLM_ON_STARTUP`、`OLLAMA_*`、`CORS_ORIGINS`、`MODEL_PROVIDER`。
  > `MODEL_PROVIDER` 是**刻意不接**的：`create_chat_model` 里硬编码 `model_provider="openai"`
  > （走 DeepSeek 的 OpenAI 兼容端点）是唯一实测跑通的组合，改成 `deepseek` 会走
  > `langchain-deepseek` 另一条分支，参数名/鉴权/base_url 处理都不同。
- **没有多轮对话**：`/api/qa/ask` 是一轮一问，请求体不带 `history` / `session_id`。
  要做多轮得先想清楚历史怎么存、以及历史进 prompt 后引用编号怎么和证据编号共存。
- **没有接入 `ENABLE_LLM_JUDGE`**：目前只有「检索侧阈值 + 模型自报不足」两层防护，
  没有用另一个模型去判答案是否有据。
- **没有修改文档的接口**：改主题 / 生效日期只能删掉重新入库（因为这两列同时冗余在 Milvus 里，
  见「Milvus 集合结构」）。
- **cross-encoder 重排已实现**（`bge-reranker-v2-m3` + 逐对 sigmoid + 阈值拒答），
  但它的调参前提仍然很弱：`MIN_EVIDENCE_SCORE=0.25` 和 `RRF_K=60` 一样是**启发式起点、
  没有离线评测支撑**，拿到真实语料后应该按需调。另外：
  - **CPU 延迟**：重排要对每对 `(query, 全文)` 跑一次完整前向，实测约 0.1~0.2 秒/对；
    候选池默认封在 `RETRIEVAL_TOP_K=8` 对，所以大约多 1~1.6 秒
  - **冷启动**：首次重排请求要加载 ~2.2G 权重（本机实测 6.3 秒）。`WARMUP_ON_STARTUP`
    仍未接线，运维上靠部署后先打一次 `/search` 或 `/qa/ask` 预热
  - **首次问答还有额外 ~3.7 秒**：`create_chat_model` 第一次调用要付 langchain-openai 的
    import + 客户端初始化成本（实测 3680ms，之后每次 0ms）。**这是每进程一次的**，
    所以冷进程的第一次 `/api/qa/ask` 大约 6.3 + 3.7 + 检索 + LLM ≈ 12 秒以上。
    给 `create_chat_model` 加 `lru_cache` **不会**改善首个请求（成本照样要付），
    真正的解法是接上预热钩子
  - **LLM 自身耗时**：实测一次问答的 `llm_ms` 约 1.9~2.4 秒，其中相当一部分是**推理 token**
    （`deepseek-v4-flash` 会输出 reasoning token，实测一次 156~353 个），不是网络慢
  - **`QA_TEMPERATURE=0.2` 也是启发式起点**：没有离线评测支撑，理由只是「政策问答要贴原文」
  - **常驻内存**：float32 权重 ~2.2G，叠加在 embedding 模型之上；开发机没问题，小容器要留意
  - **`RERANKER_MODEL` 用了 `local_files_only=True`**：权重必须在本地缓存里，
    否则**快速失败**并提示怎么补，而不是慢慢去联网重试
- **BM25 索引是进程内缓存**：多 worker（`uvicorn --workers N`）下不共享，且
  `scripts/reindex_milvus.py` 是独立进程、可能改变 BM25 语料（把 `failed` 文档修成 `ready`）
  ——**reindex 之后要重启 API**。语料大到十万块量级时，首次构建应挪到后台任务。
- **`database.py` 的 `echo=True`**：会打印所有 SQL（含参数），开发期有用，建议改成 `echo=settings.debug`。
- **上传接口是 `async def` 却做阻塞 I/O**：`write_bytes()` 和同步 `Session` 操作会阻塞事件循环。
  当前单机开发可接受，将来可用 `anyio.to_thread.run_sync` 包一层。
- ~~**缺少 CORS 中间件**~~：**已修复**。`factory.py` 已挂 `CORSMiddleware`，
  白名单读 `settings.resolved_cors_origins`（把 `.env` 里的 `CORS_ORIGINS` 按逗号拆开）。
  注意 `vite.config.ts` 的 `strictPort` 不要关：端口一旦从 5173 漂走，Origin 就对不上白名单，
  浏览器只会丢一个很含糊的 CORS 报错。
- **流式问答的 LLM 阶段失败只能是 200 + `error` 事件**：`/api/qa/ask/stream` 里
  `prepare_answer` 放在 `return StreamingResponse(...)` 之前调用，所以检索阶段的
  503（没配 key）/ 400（过滤表达式非法）仍然保持正常状态码；但响应头发出之后才失败的
  大模型调用改不了状态码，只能发一个 `error` 事件（前端据此出错误条 + 重试）。
  非流式的 `/api/qa/ask` 不受影响，那条路仍然是 502。
- **无数据库迁移工具**：表结构变更只能手工 `ALTER TABLE`，建议后续引入 Alembic。
- **`app/Schemas.py` 文件名未遵循 PEP 8**（应为 `schemas.py`），改名需同步所有 import。
- **`app/` 各子目录缺少 `__init__.py`**：目前依赖 namespace package 正常工作，加文件可改善
  IDE 提示与打包兼容性。
