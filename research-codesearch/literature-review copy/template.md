---
title: Review Paper - Hierarchical Mixture-of-Experts (HMOE)
description: Rangkuman detail paper HMOE untuk Unified Code Search (Elsevier, 2026).
head:
  - - meta
    - name: keywords
      content: AI, Code Search, Mixture of Experts, NLP
---

# Hierarchical Mixture-of-Experts model for unified code search

| Informasi Jurnal | Peringkat (SCImago) |
| :--- | :---: |
| **Penulis:** Zexu Lin, Xu Zhang, et al.<br>**Jurnal:** Information Fusion 125 (Jan 2026)<br>**Penerbit:** Elsevier<br>**Topik:** Code Search, Multimodal Learning<br>**Metode Utama:** Hierarchical Mixture-of-Experts (HMOE)<br>**Tautan:** [10.1016/j.inffus.2025.103489](https://doi.org/10.1016/j.inffus.2025.103489) | [![SCImago Journal & Country Rank](https://www.scimagojr.com/journal_img.php?id=19300156903)](https://www.scimagojr.com/journalsearch.php?q=19300156903&tip=sid&exact=no) |


## 1. Pendahuluan & Masalah

*Code search* (pencarian kode) adalah tugas vital dalam rekayasa perangkat lunak modern di mana pengembang mencari potongan kode menggunakan kueri bahasa alami (*natural language*).

### Tantangan Utama
Metode yang ada saat ini, seperti model *deep learning* monolitik (contoh: CodeBERT awal), menghadapi kendala:
1.  **Skalabilitas:** Sulit menangani repositori skala besar secara efisien.
2.  **Kesenjangan Semantik:** Kesulitan menangkap hubungan rumit antara ekspresi bahasa alami dan semantik kode sumber.
3.  **Generalisasi:** Model sering kesulitan beradaptasi lintas bahasa pemrograman dan domain yang berbeda.

::: tip Solusi yang Diusulkan
Paper ini memperkenalkan **HMOE (Hierarchical Mixture-of-Experts)**, sebuah arsitektur berjenjang yang menyeimbangkan kompleksitas komputasi dengan akurasi pengambilan (*retrieval*) menggunakan mekanisme *Macro* dan *Micro* experts.
:::

## 2. Metodologi: Arsitektur HMOE

HMOE terdiri dari dua komponen utama yang bekerja secara sinergis dalam kerangka kerja terpadu:

### A. Macro Mixture-of-Experts (Macro-MoE)
* **Fungsi:** Menangkap pola semantik global tingkat tinggi.
* **Mekanisme:** Menggunakan strategi *prefix knowledge*.
    1.  Model dilatih pada tugas klasifikasi kode (sumber) untuk mendapatkan representasi spesifik bahasa (Python, Java, dll.).
    2.  Pengetahuan ini "dibekukan" dan disuntikkan sebagai *prefix* (awalan) saat melatih model pencarian kode.

### B. Micro Mixture-of-Experts (Micro-MoE)
* **Fungsi:** Memodelkan interaksi lokal yang lebih halus (*fine-grained*) antara kueri dan kode.
* **Mekanisme:** Menggunakan **Pyramid Dilated Convolutions** (konvolusi dengan berbagai skala).
    * Ini memungkinkan model melihat kode dari berbagai perspektif (*multi-view*)—baik konteks sempit maupun luas—dan menggabungkannya menjadi representasi kode yang komprehensif.

### C. Fungsi Objektif
Model dilatih menggunakan **Contrastive Loss Function** untuk memaksimalkan kemiripan antara pasangan kueri-kode yang relevan dan menjauhkan pasangan yang tidak relevan.

## 3. Detail Pengujian (*Experimental Setup*)

### Dataset
* **Sumber:** CodeSearchNet.
* **Cakupan:** 6 Bahasa Pemrograman (Ruby, JavaScript, Go, Python, Java, PHP).
* **Distribusi:** Penulis menggunakan **IID Sampler** untuk menyeimbangkan data antar bahasa guna mengurangi bias distribusi.
* **Ukuran Data:** Bervariasi dari ~24k (Ruby) hingga ~251k (Python).

### Konfigurasi Perangkat Keras
Pengujian dilakukan menggunakan spesifikasi tinggi untuk menjamin reproduktibilitas:
* **CPU:** 2x Intel Xeon Gold 6142 (2.7 GHz)
* **RAM:** 128 GB
* **GPU:** 4x Nvidia RTX3090 (24GB VRAM)

### Parameter Pelatihan
| Parameter | Nilai |
| :--- | :--- |
| Learning Rate | 1e-5 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Epochs | 10 |
| Micro-MoE Channels | 64, 128, 256 |

### Baseline Pembanding
* CodeBERT & GraphCodeBERT
* CodeT5
* UniXcoder
* **CoCoSoDa** (Baseline utama/backbone)

## 4. Hasil Eksperimen

### A. Perbandingan Kinerja (Overall MRR)
HMOE menunjukkan peningkatan kinerja yang signifikan dibandingkan model *state-of-the-art* lainnya.

| Model | Skor MRR Keseluruhan |
| :--- | :--- |
| UniXcoder | 49.2 |
| CoCoSoDa | 73.80 |
| **HMOE** | **79.18** |

> **Catatan:** HMOE mencapai skor tertinggi di semua kategori bahasa pemrograman individu (contoh: Go mencapai MRR 91.12).

### B. Studi Ablasi (*Ablation Study*)
Penulis menguji model dengan menghapus komponen tertentu untuk memvalidasi kontribusinya:

1.  **Tanpa Macro-MoE:** Skor turun ke **78.48**.
    * *Kesimpulan:* Prefix knowledge global sangat penting.
2.  **Tanpa Micro-MoE:** Skor turun ke **79.01**.
    * *Kesimpulan:* Representasi multi-view lokal memberikan detail tambahan yang krusial.
3.  **HMOE Lengkap:** Skor tertinggi **79.18**.

### C. Generalisabilitas
Teknik HMOE diterapkan pada model lain untuk menguji fleksibilitasnya:
* **GraphCodeBERT** meningkat dari 68.31 $\rightarrow$ **68.66**
* **UniXcoder** meningkat dari 74.86 $\rightarrow$ **75.48**

### D. Analisis Visual (t-SNE)
Visualisasi *embedding* menunjukkan bahwa HMOE:
* Memisahkan distribusi antar bahasa pemrograman dengan lebih jelas.
* Mengelompokkan kode yang memiliki kemiripan semantik dengan lebih rapat (kluster yang lebih padat).

## 5. Kesimpulan

Penelitian ini membuktikan bahwa **Hierarchical Mixture-of-Experts (HMOE)** adalah solusi yang efektif untuk pencarian kode terpadu.

::: info Keunggulan Utama
1.  **Efisiensi Deployment:** Model terpadu (Unified) lebih hemat sumber daya dibandingkan melatih satu model untuk setiap bahasa.
2.  **Transfer Knowledge:** Memungkinkan bahasa dengan sedikit data (*low-resource*) belajar pola dari bahasa yang kaya data (*high-resource*).
3.  **Akurasi Tinggi:** Kombinasi pemahaman global (Macro) dan lokal (Micro) menghasilkan akurasi pencarian yang superior.
:::