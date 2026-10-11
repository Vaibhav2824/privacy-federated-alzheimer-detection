import { resolve } from 'node:path';

import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vitest/config';

// One HTML entry per page; the shared header, nav and footer come from src/ui/layout.ts.
const PAGES = ['index', 'demo', 'privacy', 'federation', 'results', 'integrity', 'method'];

export default defineConfig({
  // Vercel serves the site from the domain root.
  base: '/',
  plugins: [tailwindcss()],
  build: {
    target: 'es2022',
    rollupOptions: {
      input: Object.fromEntries(PAGES.map((page) => [page, resolve(__dirname, `${page}.html`)])),
    },
    // onnxruntime-web ships large wasm binaries; the warning is expected and
    // not actionable, so raise the threshold rather than leave noise in CI.
    chunkSizeWarningLimit: 1200,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/lib/**/*.ts'],
      // inference.ts is the onnxruntime boundary: createClassifier is covered
      // with a fake session, loadClassifier is a one-line call into the runtime
      // that would require downloading the real model to execute.
      exclude: ['src/lib/inference.ts'],
      thresholds: { lines: 100, functions: 100, branches: 100, statements: 100 },
      reporter: ['text', 'lcov'],
    },
  },
});
