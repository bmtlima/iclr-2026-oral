import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  output: 'static',
  integrations: [tailwind({ applyBaseStyles: false })],
  build: { format: 'directory' },
  site: 'https://iclr-2026-oral.vercel.app',
});
