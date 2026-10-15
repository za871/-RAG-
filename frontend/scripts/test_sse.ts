/**
 * SSE 解析的测试。node 24 能直接跑 .ts（原生类型擦除），不需要 jest/vitest 那一套。
 *
 * 用法：
 *   node scripts/test_sse.ts                          # 只跑合成用例，不需要后端
 *   node scripts/test_sse.ts --live "人才引进有哪些补贴政策？"   # 再打一次真实后端，按 7 字节切块
 *
 * 为什么要按 7 字节切块打真实后端：一帧会被网络切成任意大小的 chunk，切点可能正好落在
 * `\n\n` 中间、`data:` 中间、或一个中文字符的三个字节中间。不这么切一遍，
 * 「只在本地拼一整个响应体测试」是发现不了这些边界问题的。
 */

import assert from 'node:assert/strict'

import { ApiError } from '../src/api/errors.ts'
import { SseDecoder, parseFrame } from '../src/api/sse.ts'
import type { StreamEvent } from '../src/types.ts'

let passed = 0
function check(name: string, fn: () => void) {
  fn()
  passed += 1
  console.log(`  ✓ ${name}`)
}

const enc = new TextEncoder()

/** 把一段字符串按固定字节数切块喂进去，模拟真实的 chunk 边界 */
function feed(text: string, chunkSize: number): StreamEvent[] {
  const bytes = enc.encode(text)
  const decoder = new SseDecoder()
  const events: StreamEvent[] = []
  for (let i = 0; i < bytes.length; i += chunkSize) {
    events.push(...decoder.push(bytes.subarray(i, i + chunkSize)))
  }
  events.push(...decoder.flush())
  return events
}

console.log('合成用例：')

check('单帧解析出 event 名和 JSON 负载', () => {
  const evt = parseFrame('event: delta\ndata: {"text":"你好"}')
  assert.equal(evt?.event, 'delta')
  assert.deepEqual(evt?.data, { text: '你好' })
})

check('注释行（心跳）被跳过', () => {
  const evt = parseFrame(': keep-alive\nevent: delta\ndata: {"text":"x"}')
  assert.deepEqual(evt?.data, { text: 'x' })
})

check('没有 data 的帧返回 null', () => {
  assert.equal(parseFrame('event: ping'), null)
  assert.equal(parseFrame(''), null)
})

check('data: 后的单个空格不算内容', () => {
  const evt = parseFrame('event: delta\ndata: {"text":"a b"}')
  assert.deepEqual(evt?.data, { text: 'a b' })
})

check('负载不是合法 JSON 时抛 ApiError', () => {
  assert.throws(() => parseFrame('data: {oops'), ApiError)
})

check('多个帧：以 \\n\\n 为界', () => {
  const raw =
    'event: meta\ndata: {"status":"ok"}\n\n' +
    'event: delta\ndata: {"text":"一"}\n\n' +
    'event: done\ndata: {"status":"ok","answer":"一 [1]"}\n\n'
  const events = feed(raw, 4096)
  assert.deepEqual(
    events.map((e) => e.event),
    ['meta', 'delta', 'done'],
  )
})

check('切点落在 \\n\\n 中间也不会丢帧', () => {
  const raw = 'event: delta\ndata: {"text":"甲"}\n\nevent: delta\ndata: {"text":"乙"}\n\n'
  // 从 1 到 40 逐一切块，每一种切法都必须得到同样的两帧
  for (let size = 1; size <= 40; size += 1) {
    const texts = feed(raw, size).map((e) => (e.data as { text: string }).text)
    assert.deepEqual(texts, ['甲', '乙'], `chunkSize=${size} 时丢帧或串帧`)
  }
})

check('切点落在 data: 中间也不会丢帧', () => {
  const raw = 'event: delta\ndata: {"text":"北京市"}\n\n'
  for (let size = 1; size <= 45; size += 1) {
    const events = feed(raw, size)
    assert.equal(events.length, 1, `chunkSize=${size}`)
    assert.deepEqual(events[0].data, { text: '北京市' }, `chunkSize=${size}`)
  }
})

check('中文字符被切成两半也能正确还原（TextDecoder stream 模式）', () => {
  // 「引」= E5 BC 95，逐字节喂必然把一个字符劈成三段
  const raw = 'event: delta\ndata: {"text":"引进"}\n\n'
  const events = feed(raw, 1)
  assert.deepEqual(events[0].data, { text: '引进' })
})

check('\\r\\n 换行等价', () => {
  const raw = 'event: meta\r\ndata: {"status":"ok"}\r\n\r\nevent: delta\r\ndata: {"text":"z"}\r\n\r\n'
  for (let size = 1; size <= 20; size += 2) {
    assert.deepEqual(
      feed(raw, size).map((e) => e.event),
      ['meta', 'delta'],
      `chunkSize=${size}`,
    )
  }
})

check('正文里的换行被 JSON 转义，不会截断帧', () => {
  const answer = '第一行\n第二行\n\n第三段 [1]'
  const raw = `event: done\ndata: ${JSON.stringify({ status: 'ok', answer })}\n\n`
  const events = feed(raw, 5)
  assert.equal(events.length, 1)
  assert.equal((events[0].data as { answer: string }).answer, answer)
})

check('没有 \\n\\n 收尾的残帧被丢弃（SSE 规范如此，也是我们要的安全行为）', () => {
  // 规范：「文件结束时若正处于一个事件中间，未完成的事件不派发」。
  // 这里必须丢弃：半个 `data: {"text":"尾"}` 拿去 JSON.parse 只会抛错，
  // 而如果是半个 done 被当成完整结果用掉，那就更糟了——上层宁可收到「连接中断」
  const decoder = new SseDecoder()
  assert.deepEqual(decoder.push(enc.encode('event: delta\ndata: {"text":"尾"}')), [])
  assert.deepEqual(decoder.flush(), [])
})

check('截断的流：完整的帧留下，残帧丢掉', () => {
  const decoder = new SseDecoder()
  const whole = 'event: delta\ndata: {"text":"甲"}\n\n'
  const truncated = 'event: delta\ndata: {"text":"乙"'
  const events = decoder.push(enc.encode(whole + truncated))
  assert.equal(events.length, 1)
  assert.deepEqual(events[0].data, { text: '甲' })
  assert.deepEqual(decoder.flush(), [])
})

console.log(`\n合成用例：${passed} 项通过`)

// ---------- 真实后端（可选） ----------

const liveIndex = process.argv.indexOf('--live')
if (liveIndex === -1) {
  console.log('\n（加 --live "问题" 可再打一次真实后端，按 7 字节切块做端到端校验）')
} else {
  const query = process.argv[liveIndex + 1] ?? '人才引进有哪些补贴政策？'
  const url = 'http://127.0.0.1:8000/api/qa/ask/stream'

  console.log(`\n真实后端（chunk 固定 7 字节）：${query}`)
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  assert.equal(response.status, 200, `HTTP ${response.status}`)

  const reader = response.body!.getReader()
  const decoder = new SseDecoder()
  const seen: string[] = []
  let deltaText = ''
  let doneBody: { status: string; answer: string; citations: unknown[] } | null = null

  const handle = (events: StreamEvent[]) => {
    for (const evt of events) {
      seen.push(evt.event)
      if (evt.event === 'delta') deltaText += evt.data.text
      if (evt.event === 'done') doneBody = evt.data as typeof doneBody
    }
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    // 逐块再切一遍：7 字节一块，保证切点落在所有可能的位置上
    for (let i = 0; i < value.length; i += 7) {
      handle(decoder.push(value.subarray(i, i + 7)))
    }
  }
  handle(decoder.flush())

  assert.ok(doneBody !== null, '没有收到 done')
  assert.equal(seen[seen.length - 1], 'done', 'done 不是最后一帧')
  assert.equal(seen.filter((e) => e === 'done').length, 1, 'done 出现了多次')
  if (seen.includes('meta')) {
    assert.equal(seen.filter((e) => e === 'meta').length, 1, 'meta 出现了多次')
    assert.ok(seen.indexOf('meta') < seen.indexOf('delta'), 'meta 不在 delta 之前')
  }

  const body = doneBody as unknown as { status: string; answer: string; citations: unknown[] }
  console.log(`  事件序列  : ${seen.filter((e, i) => e !== 'delta' || i < 3).slice(0, 6).join(' → ')} … 共 ${seen.length} 帧`)
  console.log(`  delta 数  : ${seen.filter((e) => e === 'delta').length}`)
  console.log(`  status    : ${body.status}`)
  console.log(`  citations : ${body.citations.length}`)
  console.log(`  累积正文  : ${deltaText.length} 字`)

  if (body.status === 'ok') {
    assert.equal(
      deltaText.trim(),
      body.answer.trim(),
      '按 7 字节切块累积出来的正文和 done.answer 不一致',
    )
    console.log('  ✓ 7 字节切块下的累积正文与 done.answer 完全一致')
  }
  console.log('\n真实后端校验通过')
}
