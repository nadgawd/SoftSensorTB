import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const target = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'

// timeout/proxyTimeout are disabled so long-running SSE chat streams are not
// severed by the dev proxy; the backend enforces its own limits.
const backend = {
  target,
  changeOrigin: true,
  timeout: 0,
  proxyTimeout: 0,
}

const proxy = {
  '/chat': backend,
  '/datasets': backend,
  '/project': backend,
  '/health': backend,
  '/api': backend,
}

// Tailscale Funnel/Serve forward requests with the machine's *.ts.net name as Host.
const allowedHosts = ['.ts.net']

export default defineConfig({
  // GitHub Pages serves a project site under /<repo>/; the deploy workflow sets it.
  base: process.env.VITE_BASE || '/',
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy, allowedHosts },
  preview: { port: 4173, proxy, allowedHosts },
})
