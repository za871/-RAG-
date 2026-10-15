from dataclasses import dataclass,field
import re

HEADING_PATTERN = re.compile(
    r"(?m)^((?:第[一二三四五六七八九十百]+[章节条])\s*.*|(?:[一二三四五六七八九十]+)、\s*.*|\d+、\s*.*)"
)


@dataclass
class ChunkDraft:
    content:str
    section:str
    page: int | None
    is_parent:bool
    parent_index:int | None = None
    metadata:dict = field(default_factory=dict)


def _windows(text:str,
               chunk_size:int = 500,
               overlap:int = 100,
               ) -> list[str]:
    text=text.strip()
    if not text:
        return []
    chunks:list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text),start + chunk_size)
        if end < len(text):
            punctuation = max(text.rfind(mark,start,end) for mark in "。；！？\n")
            if punctuation>start+chunk_size // 2:
                end=punctuation+1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start+1,end-overlap)
    return [chunk for chunk in chunks if chunk]

def _group_sections(
    sections: list[tuple[str | None, str]],
    max_size: int,
) -> list[list[tuple[str | None, str]]]:
    """把连续小节累积成父块分组，每组总长尽量不超过 max_size。

    单个小节本身就超长时它会独占一组（由调用方按窗口再切）——凭这一点，
    「一组 = 一个父块」在正常路径下才成立，父块才稳定地比子块大。
    """
    groups: list[list[tuple[str | None, str]]] = []
    current: list[tuple[str | None, str]] = []
    current_len = 0
    for section in sections:
        if current and current_len + len(section[1]) > max_size:
            groups.append(current)
            current = []
            current_len = 0
        current.append(section)
        current_len += len(section[1])
    if current:
        groups.append(current)
    return groups


def split_policy_text(
    text: str,
    page: None,
    parent_size: int = 1200,
    child_size: int = 420,
    child_overlap: int = 80
) -> list[ChunkDraft]:
    match_list = list(HEADING_PATTERN.finditer(text))
    if not match_list:
        sections = [(None, text)]
    else:
        sections = []
        # 提取标题前的前置文本
        if match_list[0].start() > 0:
            content = text[0 : match_list[0].start()]
            sections.append( (None, content) )
        # 遍历所有标题，切割每个小节
        for index, match in enumerate(match_list):
            if index + 1 < len(match_list):
                next_title_start = match_list[index+1].start()
            else:
                next_title_start = len(text)
            content = text[match.start() : next_title_start]
            sections.append( (match.group(1), content) )

    drafts: list[ChunkDraft] = []
    # 父块按「累积多个小节」聚合，而不是一个小节一个父块。
    # 政策文件里 第X条/一、/1、 都只是一两行，按小节切的话父块和子块一样长
    # （1200 和 420 的窗口都装得下），small-to-big 等于没做
    for group in _group_sections(sections, parent_size):
        group_text = "".join(body for _, body in group)

        if len(group_text) <= parent_size:
            # 正常路径：一组小节合成一个父块（≈1200 字），子块按小节切（≈420 字）。
            # 父块横跨多个小节、子块只落在一个小节里，这个落差就是 small-to-big 的全部意义
            drafts.append(ChunkDraft(
                content=group_text.strip(),
                section=group[0][0],
                page=page,
                is_parent=True,
            ))
            # 子块按小节分别切，section 标签才停在「第X条」这一级，而不是整段父块共用一个
            for sec_title, sec_body in group:
                for child_draft in _windows(sec_body, child_size, child_overlap):
                    drafts.append(ChunkDraft(
                        content=child_draft,
                        section=sec_title,
                        page=page,
                        is_parent=False,
                    ))
        else:
            # 单个小节就超过 parent_size：退化成「先按 parent_size 切父块，子块取自父块窗口」，
            # 唯一目的是保证父块一定装得下自己的子块
            for sec_title, sec_body in group:
                # 父块：窗口1200，重叠固定100
                for j, parent_text in enumerate(_windows(sec_body, parent_size, 100), start=1):
                    drafts.append(ChunkDraft(
                        content=parent_text,
                        section=sec_title,
                        page=page,
                        is_parent=True,
                    ))
                    for child_draft in _windows(parent_text, child_size, child_overlap):
                        drafts.append(ChunkDraft(
                            content=child_draft,
                            section=sec_title,
                            page=page,
                            is_parent=False,
                            parent_index=j
                        ))
    return drafts