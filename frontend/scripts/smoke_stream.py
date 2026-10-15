"""SSE 冒烟测试：按帧打印 /api/qa/ask/stream 的事件，并校验事件顺序。

放在 frontend/scripts/ 下是为了有个「不依赖浏览器」的验证手段：
前端解析 SSE 的那套逻辑（client.ts）改动后，可以先用它确认后端这一侧没变。

用法：python frontend/scripts/smoke_stream.py "你的问题"
"""

import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
URL = f"{BASE}/api/qa/ask/stream"


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "人才引进有哪些补贴政策？"
    payload = {"query": query}

    print(f"→ {query}")
    started = time.perf_counter()
    events: list[str] = []
    deltas = 0
    text = ""
    done_body: dict | None = None

    with httpx.stream("POST", URL, json=payload, timeout=300.0) as response:
        print(f"HTTP {response.status_code}  "
              f"content-type={response.headers.get('content-type')}")
        if response.status_code != 200:
            response.read()
            print("body:", response.text[:500])
            return 1

        event_name = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].lstrip())
                elapsed = (time.perf_counter() - started) * 1000
                events.append(event_name or "?")
                if event_name == "meta":
                    print(f"[{elapsed:7.0f}ms] meta  {json.dumps(data, ensure_ascii=False)}")
                elif event_name == "delta":
                    deltas += 1
                    text += data["text"]
                    if deltas <= 3 or deltas % 25 == 0:
                        print(f"[{elapsed:7.0f}ms] delta #{deltas} {data['text']!r}")
                elif event_name == "done":
                    done_body = data
                    print(f"[{elapsed:7.0f}ms] done")
                elif event_name == "error":
                    print(f"[{elapsed:7.0f}ms] error {json.dumps(data, ensure_ascii=False)}")

    if done_body is None:
        print("!! 没有收到 done 事件")
        return 1

    print(f"\n共 {deltas} 个 delta，累积 {len(text)} 字")
    print(f"status          = {done_body['status']}")
    print(f"citations       = {len(done_body['citations'])}"
          f"  valid={done_body['citations_valid']}")
    print(f"timings         = {done_body['timings']}")
    print(f"usage           = {done_body['debug']['usage']}")
    print(f"answer(len={len(done_body['answer'])}) = {done_body['answer'][:100]}")

    # 顺序断言：done 必须是最后一帧，meta 至多一次且在 delta 之前
    assert events[-1] == "done", f"最后一帧不是 done：{events[-1]}"
    assert events.count("done") == 1, f"done 出现 {events.count('done')} 次"
    if "meta" in events:
        assert events.index("meta") < events.index("delta") if "delta" in events else True
        assert events.count("meta") == 1

    # 流式拼起来的正文必须和非流式一致（两条路共用 finalize）
    if done_body["status"] == "ok":
        assert text.strip() == done_body["answer"].strip() or text.strip().startswith(
            done_body["answer"].strip()[:50]
        ), "流式累积的正文和 done 里的 answer 对不上"
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
