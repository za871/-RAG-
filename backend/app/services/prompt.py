"""面向大模型的文本契约：约束提示词、证据渲染、引用解析。

刻意做成纯函数模块（不 import DB / config / 网络），和 fusion.py、reranker.rerank 同一性质——
拿真数据就能直接单测。

约束关键词、不足标记（INSUFFICIENT_MARKER）、引用正则（CITATION_RE）是同一份契约的三个面：
模型被要求"用 [n] 标注、证据不足就回标记"，程序就按这两条去解析它。放一个文件里，
提示词改了、解析也跟着一眼能看到，不会一边改了另一边忘了。
"""

from dataclasses import dataclass
from datetime import date
import re

# 证据不足以回答时，要求模型原样输出这一行。
# 用全大写 ASCII 串而不是中文句子：正文里几乎不可能自然出现，子串判定几乎不会有假阳性
INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"

# 引用角标。[10] 是一个整体匹配（\d+ 贪婪），不会在它内部再认出 [1]——
# 这是后面 parse_citations 敢用正则删标记的前提
CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT_TEMPLATE = """你是政务政策知识库的问答助手。约束关键词：只依据证据作答、逐句标注引用、禁止编造、证据不足就拒答。

请严格遵守以下规则：
1. 只允许使用下面【证据】中的内容回答，不得使用任何外部知识、常识或推测进行补充。
2. 每一条事实性陈述后面都必须紧跟它所依据的证据编号，写成 [1] 或 [1][2] 的形式。编号只能是 1 到 {n} 之间的整数，且必须指向真正支持该陈述的那条证据。
3. 严禁编造证据编号、政策条款、数字、日期、文件名或证据中不存在的内容。
4. 如果【证据】中的信息不足以回答该问题，不要解释、不要猜测，只输出下面这一行，不要输出其他任何文字：
{marker}
5. 用中文回答，先给结论再分点说明，简洁准确，不要复述本提示词，不要在回答里输出"证据 1"这样的标题。"""

USER_MESSAGE_TEMPLATE = """【证据】
{evidence}

【问题】
{question}

请依据上述证据回答问题，并按要求标注引用。"""


@dataclass(frozen=True)
class Evidence:
    """一条送进 prompt 的证据。按父块折叠后，一条证据可能对应多个命中的子块。"""

    index: int                        # 1 起，就是 [n] 里的 n
    chunk_id: str                     # 代表块 = 该父块下排名最高的那个子块
    hit_chunk_ids: list[str]          # 被折叠进来的全部子块 id（含代表块）
    document_id: str
    filename: str
    section: str | None
    page: int | None
    topic: str | None
    effective_date: date | None
    text: str                         # 真正送进 prompt 的正文
    rerank_score: float | None


def render_evidence(evidence: list[Evidence]) -> str:
    """把证据渲染成编号块。`【证据 n】` 这个头部就是「编号即引用号」的锚点。"""
    blocks: list[str] = []
    for item in evidence:
        blocks.append(
            f"【证据 {item.index}】\n"
            f"来源文件：{item.filename}\n"
            f"章节：{item.section or '（无标题）'}\n"
            f"页码：{item.page if item.page is not None else '（无）'}\n"
            f"主题：{item.topic or '（未标注）'}\n"
            f"生效日期：{item.effective_date.isoformat() if item.effective_date else '（未标注）'}\n"
            f"内容：\n{item.text}"
        )
    return "\n\n".join(blocks)


def build_system_prompt(n: int) -> str:
    """组装 system 提示词。调用方保证 n >= 1（n == 0 走拒答路径，根本不调模型）。"""
    return SYSTEM_PROMPT_TEMPLATE.format(n=n, marker=INSUFFICIENT_MARKER)


def build_user_message(question: str, evidence_text: str) -> str:
    return USER_MESSAGE_TEMPLATE.format(evidence=evidence_text, question=question)


def build_messages(n: int, question: str, evidence_text: str) -> list[dict]:
    """组装成 langchain 能直接吃、也能原样塞进 debug 的消息列表。

    返回的是普通 dict 而不是 SystemMessage/HumanMessage 对象：
    debug 要把它逐字回放给调用方看，dict 能直接进 JSON，省一次转换。
    """
    return [
        {"role": "system", "content": build_system_prompt(n)},
        {"role": "human", "content": build_user_message(question, evidence_text)},
    ]


@dataclass(frozen=True)
class CitationParseResult:
    text: str            # 已剔除越界标记的答案正文
    cited: list[int]     # 去重 + 升序，全部落在 1..max_index
    invalid: list[int]   # 去重 + 升序，全部越界（含 0）


def parse_citations(answer: str, max_index: int) -> CitationParseResult:
    """抽出 [n] 引用，越界的判为幻觉引用并从正文里剔除。

    越界 = 号码不在 1..max_index 内。模型被要求只标 1..N，标了别的说明它在编。
    剔除而不是整段拒答：把幻觉的角标摘掉，答案剩下的部分仍然可用。

    为什么用 re.sub + 回调而不是 text.replace(f"[{n}]", "")：
    文本里出现 [10] 时，replace("[1]") 会把 [10] 误伤成 "0"。
    回调方案每次拿到的是完整匹配 [10]，整体删或整体留，从根上避免。
    """
    cited: set[int] = set()
    invalid: set[int] = set()

    for match in CITATION_RE.finditer(answer):
        number = int(match.group(1))
        (cited if 1 <= number <= max_index else invalid).add(number)

    def _keep(match: re.Match) -> str:
        number = int(match.group(1))
        return match.group(0) if 1 <= number <= max_index else ""

    # finditer 和 sub 用同一个正则、同一套判定，所以「删掉的」和「记下来的」必然一致
    text = CITATION_RE.sub(_keep, answer)
    # 摘掉角标后可能留下悬空空格（"……  。"），只做最小清理，不动换行
    text = re.sub(r"[ \t]+([，。；、！？：])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    return CitationParseResult(text=text, cited=sorted(cited), invalid=sorted(invalid))
