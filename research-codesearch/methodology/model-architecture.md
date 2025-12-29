# Arsitektur Model: Bi-Encoder

Inti dari sistem pencarian kode ini adalah arsitektur **Bi-Encoder** (atau sering disebut *Siamese Network*). Pendekatan ini dipilih karena efisiensinya dalam pencarian skala besar.

## 1. Backbone Model (CodeBERT)

Sebagai dasar (*backbone*), kami menggunakan **CodeBERT** (dan variannya seperti GraphCodeBERT). Ini adalah model Transformer pra-latih (*pre-trained*) yang memiliki arsitektur 12-layer (mirip dengan RoBERTa-base).

*   **Input:** Mendukung input bimodal (Bahasa Alami + Bahasa Pemrograman).
*   **Hidden Size:** 768 dimensi.
*   **Parameters:** ~125 Juta parameter.

## 2. Strategi Bi-Encoder

Alih-alih memasukkan Query dan Kode secara bersamaan ke dalam satu model (*Cross-Encoder*) yang lambat, kami memisahkan pemrosesan keduanya:

1.  **Query Encoder ($E_Q$):** Mengubah input teks bahasa alami (pertanyaan user) menjadi vektor representasi $v_q$.
2.  **Code Encoder ($E_C$):** Mengubah snippet kode menjadi vektor representasi $v_c$.

Kedua encoder ini berbagi bobot (*shared weights*) yang sama untuk memastikan mereka memetakan input ke ruang vektor semantik yang sama.

### Skema Arsitektur

```text
       [Query]                     [Code Snippet]
          |                              |
    +-----------+                  +-----------+
    |  Encoder  | <--- Shared ---> |  Encoder  |
    | (CodeBERT)|      Weights     | (CodeBERT)|
    +-----------+                  +-----------+
          |                              |
          v                              v
    [Vektor q]                     [Vektor c]
          \                             /
           \                           /
            -----> [Similarity] <-----
                     (Dot Product)