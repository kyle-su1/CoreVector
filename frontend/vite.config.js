import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the FastAPI BFF so the browser sees a single origin (no CORS).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1", // bind IPv4 explicitly — browsers reach localhost over IPv4 here
    port: 5180,
    strictPort: true, // fail loudly instead of silently floating into a port collision
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
