import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy /api/* to FastAPI during development (npm run dev)
    proxy: {
      "/api": "http://localhost:8001",
    },
  },
});
