import { defineConfig } from "vitest/config";

export default defineConfig({
  cacheDir: "../node_modules/.vite/discojs-web",
  test: {
    environment: "jsdom",
    // node-datachannel#222 triggers some segfaults
    pool: "forks",
  },
});
