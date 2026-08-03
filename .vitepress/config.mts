import { defineConfig } from "vitepress";
import SIDEBAR_PPP_V_1_0 from "./sidebar-book-ppp-v1.js";
import SIDEBAR_BDD_V_1_0 from "./sidebar-book-bdd-v1.js";
import SIDEBAR_RESEARCH_CODE_SEARCH from "./sidebar-research-code-search.js";

export default defineConfig({
  title: "MRF",
  description: "M Rizki Fadhilah",

  // Aktifkan MathJax untuk rumus matematika riset
  markdown: {
    math: true,
    config: (md) => {
      // Rule untuk menangkap bracket dan membungkusnya dengan span
      md.inline.ruler.after('text', 'bracket_opacity', (state, silent) => {
        const max = state.posMax;
        const start = state.pos;

        // Cek karakter pembuka '['
        if (state.src.charCodeAt(start) !== 91) return false;

        let level = 1;
        let pos = start + 1;

        // Cari pasangan penutup ']'
        while (pos < max) {
          const code = state.src.charCodeAt(pos);
          if (code === 91) level++;
          if (code === 93) {
            level--;
            if (level === 0) break;
          }
          pos++;
        }

        if (level !== 0 || silent) return false;

        // --- PEMBUATAN TOKEN ---
        
        // 1. Token pembuka span
        const tokenOpen = state.push('span_open', 'span', 1);
        tokenOpen.attrs = [['class', 'bracket-dim']];

        // 2. Tandai posisi awal setelah '['
        state.pos++; 
        
        // 3. Tambahkan '[' sebagai teks manual (opsional, jika ingin kurungnya ikut redup)
        const tOpen = state.push('text', '', 0);
        tOpen.content = '[';

        // 4. Perintahkan markdown-it untuk memproses konten di dalam bracket
        // Kita gunakan tokenize untuk memproses konten inline (bold, italic, dll)
        const oldPosMax = state.posMax;
        state.posMax = pos;
        state.md.inline.tokenize(state);
        state.posMax = oldPosMax;

        // 5. Tambahkan ']' sebagai teks manual
        const tClose = state.push('text', '', 0);
        tClose.content = ']';

        // 6. Token penutup span
        state.push('span_close', 'span', -1);

        // 7. Geser posisi state ke setelah ']'
        state.pos = pos + 1;
        return true;
      });
    }
  },

  themeConfig: {
    outline: {
      level: [2, 3], // Ini akan menampilkan h1, h2, dan h3
      label: 'On this page' // Opsional: mengubah teks judul sidebar
    },
    nav: [
      // { text: 'Home', link: '/' },
      // { text: 'Metodologi', link: '/methodology/' },
      // { text: 'Eksperimen', link: '/experiments/' }
    ],

    sidebar: {
      "/course-bdd/202601/": [],

      "/course-ppp/202501/": [],
      "/course-ppp/202601/": [],
      "/course-ppp/202602/": [],

      "/course-bdd/202501/": [
        {
          text: "Bab 9 - Konsep dan Arsitektur IoT serta Embedded Systems",
          collapsed: true, // Bisa di-collapse biar rapi
          items: [
            { text: "00 - RPS", link: "/00-rps/" },
            { text: "01 - W1 - 0", link: "/01-week01/0" },
            { text: "01 - W1 - 1", link: "/01-week01/1" },
            { text: "01 - W1 - 2", link: "/01-week01/2" },
            { text: "01 - W1 - 3", link: "/01-week01/3" },
            { text: "01 - W1 - 4", link: "/01-week01/4" },
            { text: "01 - W1 - 5", link: "/01-week01/5" },
            { text: "01 - W1 - 6", link: "/01-week01/6" },

            { text: "02 - W2", link: "/02-week02/0" },
            { text: "03 - W3", link: "/03-week03/0" },
            { text: "04 - W4", link: "/04-week04/0" },
            { text: "05 - W5", link: "/05-week05/0" },
          ],
        },
        {
          text: "Bab 10 - Manajemen Data: Lokal, Server, dan Sinkronisasi",
          collapsed: true, // Bisa di-collapse biar rapi
          items: [
            { text: "Bab 10", link: "/bab-10/0" },
            { text: "Bab 10.1", link: "/bab-10/1" },
            { text: "Bab 10.2", link: "/bab-10/2" },
            { text: "Bab 10.3", link: "/bab-10/3" },
            { text: "Bab 10.4", link: "/bab-10/4" },
            { text: "Bab 10.z", link: "/bab-10/z" },
          ],
        },
        {
          text: "Bab 11 - Integrasi Konsep Lanjutan (Network & Device Access)",
          collapsed: true, // Bisa di-collapse biar rapi
          items: [
            { text: "Bab 11", link: "/bab-11/0" },
            { text: "Bab 11.1", link: "/bab-11/1" },
            { text: "Bab 11.2", link: "/bab-11/2" },
            { text: "Bab 11.3", link: "/bab-11/3" },
            { text: "Bab 11.4", link: "/bab-11/4" },
            { text: "Bab 11.z", link: "/bab-11/z" },
          ],
        },
        {
          text: "Bab 12 - Strategi Pengembangan Cross-Platform dan Native",
          collapsed: true, // Bisa di-collapse biar rapi
          items: [
            { text: "Bab 12", link: "/bab-12/0" },
            { text: "Bab 12.1", link: "/bab-12/1" },
            { text: "Bab 12.2", link: "/bab-12/2" },
            { text: "Bab 12.3", link: "/bab-12/3" },
            { text: "Bab 12.4", link: "/bab-12/4" },
            { text: "Bab 12.z", link: "/bab-12/z" },
          ],
        },
        {
          text: "Bab 13 - Prinsip Keamanan Platform",
          collapsed: true, // Bisa di-collapse biar rapi
          items: [
            { text: "Bab 13", link: "/bab-13/0" },
            { text: "Bab 13.1", link: "/bab-13/1" },
            { text: "Bab 13.2", link: "/bab-13/2" },
            { text: "Bab 13.3", link: "/bab-13/3" },
            { text: "Bab 13.4", link: "/bab-13/4" },
            { text: "Bab 13.z", link: "/bab-13/z" },
          ],
        },
        {
          text: "Bab 14 - Optimasi Kinerja dan Teknik Debugging Lanjut",
          collapsed: true, // Bisa di-collapse biar rapi
          items: [
            { text: "Bab 14", link: "/bab-14/0" },
            { text: "Bab 14.1", link: "/bab-14/1" },
            { text: "Bab 14.2", link: "/bab-14/2" },
            { text: "Bab 14.3", link: "/bab-14/3" },
            { text: "Bab 14.4", link: "/bab-14/4" },
            { text: "Bab 14.z", link: "/bab-14/z" },
          ],
        },
        {
          text: "Bab 15 - Trend & Integrasi Konsep: Studi Kasus Multiplatform",
          collapsed: true, // Bisa di-collapse biar rapi
          items: [
            { text: "Bab 15", link: "/bab-15/0" },
            { text: "Bab 15.1", link: "/bab-15/1" },
            { text: "Bab 15.2", link: "/bab-15/2" },
            { text: "Bab 15.3", link: "/bab-15/3" },
            { text: "Bab 15.4", link: "/bab-15/4" },
            { text: "Bab 15.z", link: "/bab-15/z" },
          ],
        },
      ],

      "/book-ppp/v-1-0/": [...SIDEBAR_PPP_V_1_0],
      "/book-bdd/v-1-0/": [...SIDEBAR_BDD_V_1_0],
      "/research-codesearch/": [...SIDEBAR_RESEARCH_CODE_SEARCH],
    },

    socialLinks: [{ icon: "github", link: "https://github.com/username/repo-riset" }],
  },
});
