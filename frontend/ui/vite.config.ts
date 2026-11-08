import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    global: 'globalThis',
  },
  resolve: {
    alias: {
      process: 'process/browser',
    },
  },
  build: {
    rollupOptions: {
      output: {
        // keep the framework in a stable, cacheable chunk; pdf.js travels with
        // the lazily imported PdfPane component instead of the entry chunk
        manualChunks(id) {
          if (
            id.includes('node_modules/react-dom/') ||
            id.includes('node_modules/react/') ||
            id.includes('node_modules/scheduler/')
          ) {
            return 'vendor-react';
          }
          return undefined;
        },
      },
    },
  },
  optimizeDeps: {
    include: ['process', 'buffer'],
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_API ?? 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
});
