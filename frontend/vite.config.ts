import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The deployed MiniGPT backend (Vercel). The browser never calls it directly:
// requests to /api/* go to the Vite dev server, which forwards them here
// server-side, so CORS does not apply.
//   GET /api/health  ->  GET https://minigptfromscratch.vercel.app/health
const BACKEND_URL = 'https://minigptfromscratch.vercel.app'

const apiProxy = {
  '/api': {
    target: BACKEND_URL,
    changeOrigin: true, // sends the backend's own Host header, which Vercel needs for routing
    secure: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: apiProxy,
  },
  preview: {
    proxy: apiProxy,
  },
})