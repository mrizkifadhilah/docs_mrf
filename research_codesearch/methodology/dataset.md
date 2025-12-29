# Dataset Eksperimen

Untuk melatih dan mengevaluasi model pencarian kode, riset ini menggunakan dataset standar industri, yaitu **CodeSearchNet Corpus**.

## CodeSearchNet Corpus

Dataset ini dikumpulkan oleh GitHub dan Weights & Biases sebagai bagian dari "CodeSearchNet Challenge". Dataset ini merupakan kumpulan fungsi (*methods*) yang diekstraksi dari repositori *open-source* di GitHub.

Setiap item data merupakan pasangan:
1.  **Snippet Kode** (Function/Method body)
2.  **Natural Language Query** (Diambil dari *docstring* atau komentar fungsi tersebut)
3.  

### Statistik Data

Berikut adalah distribusi data berdasarkan bahasa pemrograman yang digunakan dalam riset ini:

| Bahasa | Training | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Python** | 412,178 | 23,107 | 22,176 | ~457k |
| **Java** | 454,451 | 15,324 | 26,909 | ~496k |
| **JavaScript** | 123,889 | 8,253 | 6,483 | ~138k |
| **PHP** | 523,712 | 26,015 | 28,391 | ~578k |
| **Go** | 317,832 | 14,242 | 14,291 | ~346k |
| **Ruby** | 48,791 | 2,209 | 2,279 | ~53k |

# Test Full Width Table

| ID | Nama Depan | Nama Belakang | Email | Posisi | Departemen | Lokasi | Tanggal Masuk | Gaji (USD) | Status | Last Login | IP Address | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Budi | Santoso | budi@example.com | Frontend Dev | Engineering | Jakarta | 2023-01-15 | $5,000 | Active | 2023-10-01 | 192.168.1.1 | Perlu update laptop |
| 2 | Siti | Aminah | siti@example.com | Backend Dev | Engineering | Bandung | 2022-11-20 | $5,200 | Active | 2023-10-02 | 192.168.1.2 | Remote worker |
| 3 | Joko | Anwar | joko@example.com | PM | Product | Surabaya | 2021-05-10 | $6,000 | Cuti | 2023-09-28 | 192.168.1.3 | Sedang cuti hamil |
| 4 | Rina | Nose | rina@example.com | Designer | Design | Bali | 2023-02-01 | $4,800 | Active | 2023-10-03 | 192.168.1.4 | - |
| 5 | Asep | Sunandar | asep@example.com | QA | Engineering | Jakarta | 2020-08-15 | $4,500 | Inactive | 2023-08-01 | 192.168.1.5 | Resigned |

::: tip Catatan
Data dibagi menjadi tiga bagian:
* **Train:** Digunakan untuk melatih bobot model (fine-tuning).
* **Validation:** Digunakan untuk memantau performa saat training dan mencegah *overfitting*.
* **Test:** Digunakan untuk evaluasi akhir (menghitung metrik MRR).
:::

## Format Data

Data disimpan dalam format JSONL (*JSON Lines*). Berikut adalah contoh struktur satu sampel data:

```json
{
  "repo": "pandas-dev/pandas",
  "path": "pandas/core/generic.py",
  "func_name": "NDFrame.head",
  "original_string": "def head(self, n=5):\n    \"\"\"\n    Return the first `n` rows.\n    \"\"\"\n    return self.iloc[:n]",
  "language": "python",
  "code": "def head(self, n=5):\n    return self.iloc[:n]",
  "code_tokens": ["def", "head", "(", "self", ",", "n", "=", "5", ")", ":", "return", "self", ".", "iloc", "[", ":", "n", "]"],
  "docstring": "Return the first `n` rows.",
  "docstring_tokens": ["Return", "the", "first", "n", "rows", "."]
}