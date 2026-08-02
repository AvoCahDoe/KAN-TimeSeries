import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Set base to '/repo-name/' when deploying to GitHub Pages project sites.
export default defineConfig({
  plugins: [react()],
  base: './',
})
