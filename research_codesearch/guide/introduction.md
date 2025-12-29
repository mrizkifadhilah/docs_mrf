# Pengantar Research Code Search

Dokumentasi ini berisi catatan, temuan, dan metodologi terkait riset **Code Search** (Pencarian Kode). Bagian ini menjelaskan konsep dasar dan motivasi di balik riset ini.

## Apa itu Code Search?

Code Search adalah proses menemukan snippet kode yang relevan dari repositori kode yang besar berdasarkan kueri yang diberikan oleh pengguna. Kueri ini bisa berupa:
*   **Kata kunci (Keywords):** Contoh `quick sort java`
*   **Bahasa Alami (Natural Language):** Contoh *"how to sort list recursively in python"*

::: info Penting
Tujuan utama dari modern Code Search adalah menjembatani *semantic gap* (celah makna) antara bahasa manusia (natural language) dan bahasa pemrograman.
:::

## Motivasi Riset

Pengembangan perangkat lunak modern sangat bergantung pada penggunaan ulang kode (*code reuse*). Developer sering menghabiskan waktu untuk mencari implementasi fungsi tertentu.

1.  **Efisiensi:** Mempercepat proses development.
2.  **Kualitas:** Menemukan implementasi standar atau *best practice*.
3.  **Pembelajaran:** Membantu developer pemula memahami cara kerja algoritma.

## Klasifikasi Pendekatan

Dalam riset ini, kita akan mengeksplorasi dua pendekatan utama:

### 1. Information Retrieval (IR) Based
Pendekatan tradisional yang menggunakan pencocokan kata kunci (*keyword matching*).
*   **Metode:** TF-IDF, BM25.
*   **Kelebihan:** Cepat, mudah diimplementasikan.
*   **Kekurangan:** Tidak memahami konteks atau sinonim (misal: "array" vs "list").

### 2. Deep Learning / Semantic Based
Pendekatan modern menggunakan Neural Networks untuk memetakan kode dan kueri ke dalam ruang vektor yang sama (*embedding space*).
*   **Metode:** CodeBERT, GraphNeuralNetworks (GNN), CodeT5.
*   **Kelebihan:** Memahami semantik dan konteks.
*   **Kekurangan:** Membutuhkan resource komputasi besar untuk training.

## Roadmap Riset

Dokumentasi ini akan mencakup:
- [ ] Studi Literatur (Paper Review)
- [ ] Metodologi Pengumpulan Dataset
- [ ] Arsitektur Model
- [ ] Eksperimen & Evaluasi Hasil