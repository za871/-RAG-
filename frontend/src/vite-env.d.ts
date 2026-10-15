/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端地址，例如 http://127.0.0.1:8000。不设则用 client.ts 里的默认值 */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
