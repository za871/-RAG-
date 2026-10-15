import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 不配 server.proxy：前端走真实跨域，后端已挂 CORSMiddleware 读取 .env 的 CORS_ORIGINS。
// 这样 dev 和「把 build 产物丢到别的域名下」这两种部署跑的是同一条路径，
// 不会出现「dev 没事、上线被 CORS 拦住」。
// 后端地址通过 VITE_API_BASE 覆盖（见 src/api/client.ts）。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 端口被占时直接失败，而不是悄悄换到 5174——换了端口 Origin 就对不上 CORS 白名单了
    strictPort: true,
  },
})
