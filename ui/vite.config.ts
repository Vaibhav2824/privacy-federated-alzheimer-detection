import { defineConfig } from 'vitest/config';

// The site is served from a project page (https://<user>.github.io/<repo>/),
// so every asset URL has to be namespaced under the repository name. Vite's
// BASE_URL is read back in src/lib/inference.ts to locate the model.
const repositoryBase = '/privacy-federated-alzheimer-detection/';

export default defineConfig({
  base: process.env.GITHUB_PAGES === 'true' ? repositoryBase : '/',
  build: {
    target: 'es2022',
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
