import { defineConfig } from "vite";

const root = new URL(".", import.meta.url).pathname;

/**
 * Vite config for the interactive demos. The core library is consumed through
 * source aliasing so the demos always exercise the working tree and never a
 * stale build. Three.js is a devDependency only; nothing in `src/` references
 * it.
 */
export default defineConfig({
  root,
  resolve: {
    alias: {
      impulse3d: new URL("../src/index.ts", import.meta.url).pathname,
    },
  },
  server: {
    port: 5173,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
  },
});
