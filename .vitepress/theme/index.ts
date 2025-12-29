import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Tipe 'app' otomatis terdeteksi di sini
  }
} satisfies Theme