"""从河南工学院官网（www.hait.edu.cn）抓取**公开**的公告正文，落到 backend/data/hait_docs
供 RAG 语料使用。

抓取范围（默认）：
  - 校务公开   /xwgk.htm  共 3 页  （全部）
  - 通知公告   /tzgg.htm  共 315 页（默认只取最近 20 页，见 --pages）

分页口径：栏目的 root 页是**最新**一页，页面上的 class="Next" 链接依次指向更旧的页
（tzgg: root → 314.htm → 313.htm … → 1.htm）。所以这里跟着 Next 走 N 页，
不靠页码公式猜——那套编号是「总页数 - 页码」的反向编号，写死必错。

**为什么不下载附件**：站点的附件走 /system/_content/download.jsp?...&wbfileid=…，
但它不直接返回文件，而是回一个「附件下载」页、要求填 createimage.jsp 生成的图形验证码。
这是站点刻意设置的反自动化措施，不去绕它。所以这里统一取**正文**：
政策/通知的正文通常已经包含要点，比整条丢掉强。

为什么落到 data/hait_docs 而不是 data/uploads：uploads 是流水线的工作目录，
`DELETE /api/documents?confirm=RESET` 会把它整个清空——爬来的源文件放那里会被误删。
入库时后端自己会把副本写进 uploads，行为不受影响。

礼貌约束：默认每个请求间隔 1 秒、带 UA、只走本域、超时重试、断线不阻塞整轮。

    python scripts/crawl_hait.py --dry-run          # 只扫描不落盘，看规模
    python scripts/crawl_hait.py                    # 抓取
    python scripts/crawl_hait.py --pages 3          # 通知公告只取 3 页
    python scripts/crawl_hait.py --ingest           # 把已抓到的文件逐个入库
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import requests

BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = BACKEND_ROOT / "data" / "hait_docs"
MANIFEST = BACKEND_ROOT / "data" / "hait_manifest.json"

UA = "Mozilla/5.0 (compatible; hait-corpus-fetcher/1.0; personal RAG corpus)"

# 正文短于这个长度就认为没有实质内容（比如只剩导航栏残留），不落盘
MIN_BODY_CHARS = 30

# (主题, 栏目首页, 默认页数) —— 页数 None 表示不限（抓到没有 Next 为止）
COLUMNS = [
    ("校务公开", "https://www.hait.edu.cn/xwgk.htm", None),
    ("通知公告", "https://www.hait.edu.cn/tzgg.htm", 20),
]

MAX_NAME = 100          # 文件名保留的最大字符数（后端 DB 的 file_name 是 VARCHAR(225)）

_session = requests.Session()
_session.headers.update({"User-Agent": UA})
_delay = 1.0
_last_req = [0.0]


def fetch(url: str, retries: int = 3) -> str | None:
    """限速 + 重试的 GET，返回 utf-8 文本；失败返回 None。"""
    for attempt in range(retries):
        wait = _delay - (time.monotonic() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _session.get(url, timeout=30)
            _last_req[0] = time.monotonic()
            if resp.status_code != 200:
                print(f"    ! HTTP {resp.status_code} {url}")
                return None
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as exc:
            _last_req[0] = time.monotonic()
            if attempt == retries - 1:
                print(f"    ! {type(exc).__name__} {url}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", "", text))).strip()


def next_page_url(html: str, base: str) -> str | None:
    """页面上的「下页」链接。

    不能只写 `<a ... class="Next" ...>`：页面里还有个装饰性元素
    `<A class="next icon-arrow-a-right" onfocus="undefined">`（无 href），
    用宽松的 class 匹配会先命中它、然后 href 取不到、整个翻页就断了。
    所以遍历所有 <a> 标签，要求 class 分词后**精确含** Next，且标签里必须真有 href。
    """
    for tag in re.findall(r"<a\b[^>]*>", html, re.I):
        cls = re.search(r'class=["\']([^"\']*)["\']', tag, re.I)
        if not cls or "Next" not in cls.group(1).split():
            continue
        href = re.search(r'href=["\']([^"\']+)["\']', tag, re.I)
        if href:
            return urljoin(base, href.group(1))
    return None


def detail_links(html: str, base: str) -> list[tuple[str, str]]:
    """详情页链接。root 页用相对路径 info/...，子页用 ../info/...，一律 urljoin 归一。"""
    out, seen = [], set()
    for href, inner in re.findall(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.S | re.I):
        url = urljoin(base, href)
        if re.search(r"/info/\d+/\d+\.htm$", url) and url not in seen:
            seen.add(url)
            out.append((clean(inner), url))
    return out


def page_title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
    return clean(m.group(1)).replace("-欢迎访问河南工学院", "").strip() if m else ""


def body_text(html: str) -> str:
    """正文 div.v_news_content。正文之后紧跟投票控件和附件区，用它俩当结束标记。"""
    i = html.find("v_news_content")
    if i < 0:
        return ""
    seg = html[i:]
    for stop in ('<div id="div_vote_id"', "attachfont", "</td></tr>"):
        j = seg.find(stop)
        if j > 0:
            seg = seg[:j]
            break
    seg = re.sub(r"<br\s*/?>", "\n", seg, flags=re.I)
    seg = re.sub(r"</(?:p|div|tr|h\d)>", "\n", seg, flags=re.I)
    text = html_mod.unescape(re.sub(r"<[^>]+>", "", seg))
    text = re.sub(r"[ \t　\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def safe_name(name: str, doc_id: str, ext: str) -> str:
    """去掉路径分隔符等非法字符，并限长——后端 DB 的 file_name 是 VARCHAR(225)。"""
    stem = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip(" ._") or "untitled"
    return f"hait_{doc_id}_{stem[:MAX_NAME]}{ext}"


def doc_id_of(url: str) -> str:
    m = re.search(r"/info/(\d+)/(\d+)\.htm$", url)
    return f"{m.group(1)}-{m.group(2)}" if m else "0-0"


def crawl(dry_run: bool, pages_override: int | None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"pages": 0, "details": 0, "txt": 0, "too_short": 0, "failed": 0, "files": []}
    seen_details: set[str] = set()

    for topic, start, default_pages in COLUMNS:
        limit = pages_override if pages_override is not None else default_pages
        print(f"\n===== {topic}  {start}  （页数上限 {limit or '不限'}）")
        url, page_no = start, 0
        while url and (limit is None or page_no < limit):
            html = fetch(url)
            if html is None:
                stats["failed"] += 1
                break
            page_no += 1
            stats["pages"] += 1
            links = detail_links(html, url)
            print(f"  第 {page_no} 页 {url.rsplit('/', 1)[-1] or 'root'}：{len(links)} 条")

            for title, detail in links:
                if detail in seen_details:
                    continue
                seen_details.add(detail)
                stats["details"] += 1
                detail_html = fetch(detail)
                if detail_html is None:
                    stats["failed"] += 1
                    continue

                text = body_text(detail_html)
                if len(text) < MIN_BODY_CHARS:
                    stats["too_short"] += 1
                    continue

                fname = safe_name(title or page_title(detail_html), doc_id_of(detail), ".txt")
                if dry_run:
                    stats["txt"] += 1
                    continue
                if (OUT_DIR / fname).exists():
                    # 同名（标题重复且栏目 id 相同）不重复写，保留先到的
                    continue
                (OUT_DIR / fname).write_text(text, encoding="utf-8")
                stats["txt"] += 1
                stats["files"].append({"file": fname, "topic": topic, "title": title, "source": detail})

            url = next_page_url(html, url)

    return stats


def _existing_filenames(api: str) -> set[str]:
    """库里已有的文件名集合。

    每 POST 一次就新建一个 PolicyDocument，所以重跑必须靠文件名去重——
    否则失败重试会把已经入库的文档再插一遍，变成重复记录。
    """
    names: set[str] = set()
    page = 1
    while True:
        r = requests.get(api, params={"page": page, "size": 100}, timeout=60)
        r.raise_for_status()
        data = r.json()
        items = data.get("items") or []
        names.update(item["filename"] for item in items if item.get("filename"))
        if not items or page * 100 >= (data.get("total") or 0):
            return names
        page += 1


def ingest(topic_default: str = "河南工学院", workers: int = 6) -> None:
    """把 manifest 里抓到的文件 POST /api/documents。已在库里的按文件名跳过（可断点续跑）。

    为什么并发：每篇文档在 Milvus 侧要 delete+flush、insert+flush 两次，flush 会 seal
    segment，单线程实测 16.7s/篇（429 篇要两小时）。这些开销几乎全是**等待**而不是 CPU，
    而上传路由是同步 def、跑在 FastAPI 线程池里，所以并发 POST 能真正并行化。
    workers 不宜太大：Milvus standalone 是单机，flush 过密反而会拖慢甚至打挂它。
    """
    api = "http://127.0.0.1:8000/api/documents"
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"] if MANIFEST.exists() else []
    if not entries:
        print("manifest 为空，先跑一次抓取")
        return
    done = _existing_filenames(api)

    todo, missing, skipped = [], 0, 0
    for item in entries:
        path = OUT_DIR / item["file"]
        if not path.exists():
            missing += 1
        elif item["file"] in done:
            skipped += 1
        else:
            todo.append(item)
    print(f"库里已有 {len(done)} 篇；本次待入库 {len(todo)}，跳过 {skipped}，缺失 {missing}；并发 {workers}\n")

    ctype = {".pdf": "application/pdf",
             ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             ".txt": "text/plain"}
    lock = threading.Lock()
    counter = {"n": 0, "ok": 0, "bad": 0}

    def post(item: dict) -> None:
        path = OUT_DIR / item["file"]
        status: int | None = None
        detail = ""
        try:
            with path.open("rb") as fh:
                r = requests.post(api, files={"file": (item["file"], fh, ctype[path.suffix.lower()])},
                                  data={"topic": item.get("topic") or topic_default}, timeout=900)
            status, detail = r.status_code, r.text[:100]
        except requests.RequestException as exc:
            detail = type(exc).__name__
        with lock:
            counter["n"] += 1
            if status in (200, 201):
                counter["ok"] += 1
            else:
                counter["bad"] += 1
                print(f"  [{counter['n']}] {status} {item['file'][:58]} :: {detail}", flush=True)
            if counter["n"] % 25 == 0 or counter["n"] == len(todo):
                print(f"  ...{counter['n']}/{len(todo)}（成功 {counter['ok']} 失败 {counter['bad']}）", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(post, todo))
    print(f"\n入库完成：成功 {counter['ok']}，失败 {counter['bad']}，"
          f"跳过(已在库) {skipped}，缺失 {missing}，共 {len(entries)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="河南工学院公开公告抓取")
    ap.add_argument("--dry-run", action="store_true", help="只扫描不落盘")
    ap.add_argument("--pages", type=int, default=None, help="通知公告取最近几页（覆盖默认 20）")
    ap.add_argument("--delay", type=float, default=1.0, help="请求间隔秒数，默认 1.0")
    ap.add_argument("--ingest", action="store_true", help="把已抓取文件入库，不再抓取")
    ap.add_argument("--workers", type=int, default=6, help="入库并发数，默认 6")
    args = ap.parse_args()

    global _delay
    _delay = args.delay

    if args.ingest:
        ingest(workers=args.workers)
        return 0

    stats = crawl(args.dry_run, args.pages)
    print("\n===== 汇总 =====")
    print(f"  列表页 {stats['pages']}，详情页 {stats['details']}（去重后），失败 {stats['failed']}")
    print(f"  正文 .txt {stats['txt']}，正文过短跳过 {stats['too_short']}")
    if not args.dry_run:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  落盘 {len(stats['files'])} 个文件到 {OUT_DIR}")
        print(f"  清单 {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
