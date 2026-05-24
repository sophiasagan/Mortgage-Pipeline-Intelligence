import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward /api/* to the FastAPI backend during local dev
      '/api': {
        target:      process.env.VITE_API_URL ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite:     (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port: parseInt(process.env.PORT ?? '4173'),
    host: true,
  },
  build: {
    outDir:    'dist',
    sourcemap: true,
  },
});
