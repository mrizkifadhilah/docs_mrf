# Preprocessing & Representasi Data

Sebelum data dimasukkan ke dalam model Neural Network, serangkaian teknik pemrosesan awal dilakukan untuk mengubah teks dan kode menjadi vektor numerik.

## 1. Tokenisasi (Byte-Pair Encoding)

Model berbasis Transformer tidak memproses teks berdasarkan kata per kata (spasi), melainkan menggunakan *Subword Tokenization*. Riset ini menggunakan **Byte-Level BPE** (seperti yang digunakan pada RoBERTa dan CodeBERT).

### Mengapa Subword?
Dalam kode pemrograman, sering ditemukan penggabungan kata (contoh: `calculateMeanSquareError`). Tokenizer standar akan menganggap ini sebagai satu kata asing (*Unknown*). BPE memecahnya menjadi bagian yang bermakna:

**Contoh:**
*   Input: `calculateMeanSquareError`
*   Tokenisasi: `['calculate', 'Mean', 'Square', 'Error']`

Ini memungkinkan model memahami makna dari variabel atau fungsi yang belum pernah dilihat sebelumnya, selama model mengenali sub-kata penyusunnya.

## 2. Struktur Input (Special Tokens)

Model Transformer memerlukan penanda khusus untuk memahami batas antar kalimat.

Format input untuk satu sekuens adalah:
$$ \text{Input} = [\text{<s>}, w_1, w_2, \dots, w_n, \text{</s>}] $$

Dimana:
*   **`<s>`**: Token klasifikasi awal (sering disebut `[CLS]` pada BERT). Vektor dari token ini nantinya digunakan sebagai representasi sem