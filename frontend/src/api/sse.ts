import { ApiError } from './errors.ts'
import type { StreamEvent } from '../types.ts'

/**
 * SSE 解析。
 *
 * 单独一个模块、不碰 fetch 也不碰 import.meta.env，是为了它能脱离浏览器直接跑测试
 * （见 scripts/test_sse.ts）。这里的三件事都很容易写错，而且错了之后表现得很隐蔽：
 *   1. 一帧可能被切成任意多个 chunk，切点还可能正好落在 `\n\n` 或 `data:` 中间
 *   2. 一个中文字符占 3 个字节，chunk 边界经常把它劈成两半（TextDecoder 必须用 stream 模式）
 *   3. 答案正文里全是换行，data 必须整体 JSON 编码，否则 SSE 会在第一个换行处截断帧
 */

/** 把一帧原始文本解析成事件；解析不出内容（注释/心跳/空帧）返回 null */
export function parseFrame(raw: string): StreamEvent | null {
  let eventName = 'message'
  const dataLines: string[] = []

  for (const line of raw.split('\n')) {
    // 以冒号开头的是注释，SSE 的心跳走这条路；这里没有需要处理的语义，直接跳过
    if (line.startsWith(':')) continue
    if (line.startsWith('event:')) {
      eventName = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      // 规范里 data: 后可以跟一个空格，那个空格不算内容
      dataLines.push(line.slice('data:'.length).replace(/^ /, ''))
    }
  }

  if (dataLines.length === 0) return null

  // 多行 data 按规范用 \n 拼接。后端的 _sse 用 json.dumps 编码，正文里的换行早被转义成 \n，
  // 所以实际永远只有一行——按规范写全是为了将来有人手工调试时也能对上
  const payload = dataLines.join('\n')
  let data: unknown
  try {
    data = JSON.parse(payload)
  } catch {
    throw new ApiError(`无法解析服务端返回的事件负载：${payload.slice(0, 200)}`)
  }
  return { event: eventName, data } as StreamEvent
}

/**
 * 字节流 → 事件的增量解码器。
 *
 * 喂进来的 chunk 边界完全是任意的，所以这里自己攒缓冲区：只有见到完整的 `\n\n`
 * 才认为一帧到齐了，剩下的留在 buffer 里等下一段字节。
 */
export class SseDecoder {
  private buffer = ''
  // stream: true 让 TextDecoder 自己处理「多字节字符被切断」的情况：
  // 它会把半个字符留在内部，等下一段字节补齐再吐出来
  private decoder = new TextDecoder('utf-8')

  /** 喂一段字节，返回这段字节里已经完整的所有帧 */
  push(chunk: Uint8Array): StreamEvent[] {
    this.buffer += this.decoder.decode(chunk, { stream: true })
    return this.drain()
  }

  /** 流结束：把残留的半个字符和最后一帧吐出来 */
  flush(): StreamEvent[] {
    this.buffer += this.decoder.decode()
    return this.drain()
  }

  private drain(): StreamEvent[] {
    const events: StreamEvent[] = []
    // 统一成 \n：两种换行都见过。\r\n 正好被切在两个 chunk 之间时，
    // 下一轮这里会连同新到的 \n 一起替换掉，所以不会漏
    this.buffer = this.buffer.replace(/\r\n/g, '\n')

    let boundary = this.buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const frame = this.buffer.slice(0, boundary)
      this.buffer = this.buffer.slice(boundary + 2)
      const event = parseFrame(frame)
      if (event) events.push(event)
      boundary = this.buffer.indexOf('\n\n')
    }
    return events
  }
}
