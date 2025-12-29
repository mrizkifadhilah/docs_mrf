# Setup Eksperimen & Konfigurasi

Bagian ini merinci perangkat keras, perangkat lunak, dan *hyperparameters* yang digunakan untuk melatih dan mengevaluasi model Semantic Code Search.

## 1. Lingkungan Perangkat Keras (Hardware)

Mengingat komputasi berat yang dibutuhkan oleh model berbasis Transformer, eksperimen ini dijalankan pada infrastruktur berikut:

*   **GPU:** NVIDIA Tesla A100 (40GB VRAM) atau setara (misal: V100 32GB).
*   **CPU:** Intel Xeon Scalable Processors (16 Cores).
*   **RAM:** 64 GB System Memory.
*   **Storage:** SSD NVMe untuk mempercepat *data loading*.

## 2. Lingkungan Perangkat Lunak (Software)

Pengembangan dilakukan menggunakan ekosistem Python dengan *library* utama sebagai berikut:

| Komponen | Spesifikasi | Keterangan |
| :--- | :--- | :--- |
| **Language** | Python 3.8+ | Bahasa pemrograman utama. |
| **Framework** | PyTorch 1.10+ | Framework Deep Learning. |
| **Library** | Hugging Face Transformers | Untuk memuat model CodeBERT dan Tokenizer. |
| **Parser** | Tree-sitter | Untuk parsing AST (Abstract Syntax Tree). |
| **Tracking** | Weights & Biases (W&B) | Untuk memantau loss dan metrik saat training. |

## 3. Konfigurasi Hyperparameter

Pemilihan hyperparameter sangat mempengaruhi konvergensi model. Berikut adalah konfigurasi terbaik (*best practice*) yang digunakan dalam eksperimen ini:

*   **Optimizer:** AdamW (Adam dengan Weight Decay).
*   **Learning Rate:** $2 \times 10^{-5}$ (dengan *linear scheduler*).
*   **Batch Size:** 32 atau 64 (tergantung kapasitas VRAM).
*   **Max Sequence Length:** 256 token (mengcover sebagian besar fungsi standar).
*   **Epochs:** 10 epoch (biasanya konvergen di epoch ke-3 hingga 5).
*   **Weight Decay:** 0.01 (untuk regularisasi).
*   **Gradient Accumulation:** Digunakan jika memori GPU terbatas untuk mensimulasikan batch size yang lebih besar.

## 4. Model Baseline

Untuk mengukur efektivitas model yang diusulkan, kami membandingkannya dengan metode berikut:

1.  **BM25 (Lexical Search):**
    Pendekatan Information Retrieval klasik berbasis frekuensi kata kunci. Ini menjadi tolak ukur pencarian berbasis teks murni tanpa pemahaman semantik.

2.  **RoBERTa (General NLP):**
    Model Transformer yang dilatih hanya pada teks bahasa alami (Wikipedia, Buku), tanpa pemahaman khusus tentang kode program. Ini untuk membuktikan pentingnya pre-training pada dataset kode (CodeBERT).