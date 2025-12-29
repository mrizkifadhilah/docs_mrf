# State of the Art (SOTA)

Halaman ini merangkum perkembangan metode *Code Search*, mulai dari pendekatan berbasis frekuensi kata hingga model bahasa skala besar (LLM).

## 1. Pendekatan Tradisional (Information Retrieval)

Sebelum era *Deep Learning*, pencarian kode bergantung pada pencocokan teks literal.

### BM25 (Best Matching 25)
Ini adalah fungsi peringkat probabilistik yang digunakan oleh banyak mesin pencari kode awal (seperti Koders atau Krugle).
*   **Konsep:** Menghitung skor relevansi berdasarkan frekuensi kemunculan kata kunci kueri di dalam dokumen kode.
*   **Keterbatasan:** Gagal menangani *mismatch* kosa kata. Jika pengguna mencari "sort" tapi kode menggunakan nama fungsi "order", BM25 mungkin tidak menemukannya.

---

## 2. Era Awal Neural Code Search

Pendekatan ini mulai memetakan kode dan kueri ke dalam vektor (*embedding*).

### DeepCS (Gu et al., 2018)
Salah satu model pertama yang menggunakan teknik *Joint Embedding* (CODENN).
*   **Arsitektur:** Menggunakan LSTM (Long Short-Term Memory) bidirectional.
*   **Input:** Memproses nama fungsi, token API, dan docstring secara terpisah.
*   **Mekanisme:** Kueri ($q$) dan Kode ($c$) dipetakan ke ruang vektor yang sama. Relevansi dihitung menggunakan *Cosine Similarity*:

$$ \text{sim}(q, c) = \frac{\mathbf{v}_q \cdot \mathbf{v}_c}{\|\mathbf{v}_q\| \|\mathbf{v}_c\|} $$

---

## 3. Era Pre-trained Models (Transformers)

Terinspirasi oleh kesuksesan BERT di NLP, model-model ini dilatih pada korpus kode yang sangat besar (seperti CodeSearchNet).

### CodeBERT (Microsoft, 2020)
*   **Deskripsi:** Model bimodal yang dilatih dengan *Masked Language Modeling* (MLM) dan *Replaced Token Detection* (RTD).
*   **Kelebihan:** Sangat kuat dalam menangkap hubungan semantik antara *natural language* dan bahasa pemrograman. Dapat digunakan untuk berbagai tugas hilir (*downstream tasks*).

### GraphCodeBERT (2021)
*   **Inovasi:** Menambahkan struktur *Data Flow Graph* (DFG) ke dalam input CodeBERT.
*   **Tujuan:** Agar model memahami tidak hanya urutan token, tetapi juga aliran variabel di dalam kode.

---

## 4. Perbandingan Performa

Tabel berikut adalah ringkasan performa (MRR - Mean Reciprocal Rank) pada dataset CodeSearchNet (Python & Java):

| Model | Arsitektur | MRR (Python) | MRR (Java) |
| :--- | :--- | :--- | :--- |
| **BM25** | Probabilistic | 0.39 | 0.41 |
| **DeepCS** | LSTM/RNN | 0.52 | 0.54 |
| **CodeBERT** | Transformer (base) | 0.67 | 0.69 |
| **GraphCodeBERT**| Transformer + DFG | **0.69** | **0.71** |

> **Catatan:** Angka di atas adalah aproksimasi berdasarkan paper asli untuk memberikan gambaran peningkatan performa.