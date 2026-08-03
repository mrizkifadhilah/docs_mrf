export default [
  {
    text: "Bab 1: Pendahuluan",
    collapsed: true, // Bisa di-collapse biar rapi
    items: [
      { text: "Latar Belakang", link: "/research-codesearch/introduction/background" },
      { text: "Tujuan Riset", link: "/research-codesearch/introduction/objectives" },
    ],
  },
  {
    text: "Bab 2: Kajian Pustaka",
    collapsed: true, // Bisa di-collapse biar rapi
    items: [
      // { text: 'Ringkasan', link: '/literature-review/' },
      // { text: 'Traditional IR (BM25)', link: '/literature-review/ir-based' },
      // { text: 'Deep Learning Models', link: '/literature-review/deep-learning' },
      // { text: 'LLM & Generative', link: '/literature-review/llm-based' },
      // { text: 'TEMPLATE', link: '/literature-review/template' },
      { text: "20251219 - FPGraphCS - Early-Stage Graph Fusion with Refined Graph Neural Networks for Semantic Code Search", link: "/research-codesearch/literature-review/20251219" },
    ],
  },
  {
    text: "Bab 3: Metodologi",
    collapsed: true, // Bisa di-collapse biar rapi
    items: [
      { text: "Dataset (CodeSearchNet)", link: "/research-codesearch/methodology/dataset" },
      { text: "Preprocessing (AST)", link: "/research-codesearch/methodology/preprocessing" },
      { text: "Arsitektur Model", link: "/research-codesearch/methodology/model-architecture" },
    ],
  },
  {
    text: "Bab 4: Eksperimen",
    collapsed: true, // Bisa di-collapse biar rapi
    items: [
      { text: "Metrik Evaluasi (MRR)", link: "/research-codesearch/experiments/metrics" },
      { text: "Hasil & Diskusi", link: "/research-codesearch/experiments/results" },
    ],
  },
  {
    text: "Bab 5: Penutup",
    collapsed: true, // Bisa di-collapse biar rapi
    items: [{ text: "Kesimpulan", link: "/research-codesearch/conclusion/" }],
  },
];
