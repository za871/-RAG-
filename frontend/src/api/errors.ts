/**
 * 后端明确拒绝（HTTP 状态码非 2xx，或流中途的 error 事件）时抛这个。
 *
 * 单独一个类型是为了让界面能区分「后端说了什么」和「网络/代码炸了」：
 * 前者直接把 detail 显示给用户（那是后端写给人看的文案），后者只显示一句通用提示。
 */
export class ApiError extends Error {
  /** HTTP 状态码；流中途的 error 事件拿不到状态码，此时为 0 */
  readonly status: number

  // 不用 TS 的构造函数参数属性写法（`constructor(readonly status = 0)`）：
  // node 的原生类型擦除不支持它，而 scripts/test_sse.ts 要直接 import 这个模块跑测试
  constructor(message: string, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** 用户点了「停止生成」*/
export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}
