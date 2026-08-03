# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Abstract

Kami memperkenalkan model representasi bahasa baru yang disebut BERT, yang merupakan singkatan dari Bidirectional Encoder Representations from Transformers. Berbeda dengan model representasi bahasa terkini , BERT dirancang untuk melakukan pre-train — prapelatihan — representasi bidireksional mendalam dari teks tanpa label dengan secara bersama-sama mengondisikan konteks kiri dan kanan di seluruh lapisan. Sebagai hasilnya, model BERT yang telah dilatih sebelumnya dapat melalui proses fine-tuned — penalaan halus — hanya dengan satu lapisan output tambahan untuk menciptakan model mutakhir (state-of-the-art) bagi berbagai macam tugas, seperti menjawab pertanyaan dan inferensi bahasa, tanpa modifikasi arsitektur spesifik tugas yang substansial.

BERT secara konseptual sederhana dan secara empiris kuat. Model ini memperoleh hasil mutakhir baru pada sebelas tugas pemrosesan bahasa alami (Natural Language Processing — NLP), termasuk meningkatkan skor GLUE menjadi 80,5% (peningkatan absolut 7,7 poin persentase), akurasi MultiNLI menjadi 86,7% (peningkatan absolut 4,6%), F1 Test pada menjawab pertanyaan SQuAD v1.1 menjadi 93,2 (peningkatan absolut 1,5 poin), dan F1 Test pada SQuAD v2.0 menjadi 83,1 (peningkatan absolut 5,1 poin).

## 1 - Introduction

## 1 - Introduction

**Language model pre-training** — prapelatihan model bahasa — telah terbukti efektif untuk meningkatkan berbagai tugas pemrosesan bahasa alami [*NLP*, Dai and Le, 2015; *ELMo*, Peters et al, 2018a; *OpenAI GPT*, Radford et al, 2018; *ULMFiT*, Howard and Ruder, 2018]. Tugas-tugas ini mencakup tugas tingkat kalimat seperti **natural language inference** — inferensi bahasa alami [*SNLI*, Bowman et al, 2015; *MultiNLI*, Williams et al, 2018] dan **paraphrasing** — parafrasa [Dolan and Brockett, 2005], yang bertujuan memprediksi hubungan antar kalimat dengan menganalisisnya secara holistik, serta tugas tingkat token seperti **named entity recognition** — pengenalan entitas bernama dan **question answering** — menjawab pertanyaan, di mana model dituntut untuk menghasilkan output granular pada level token [*CoNLL-2003*, Tjong Kim Sang and De Meulder, 2003; *SQuAD*, Rajpurkar et al, 2016].

Terdapat dua strategi utama untuk menerapkan representasi bahasa yang telah dilatih sebelumnya ke tugas **downstream** — tugas hilir: **feature-based** — berbasis fitur dan **fine-tuning** — penalaan halus. Pendekatan **feature-based**, seperti [*ELMo*, Peters et al, 2018a], menggunakan arsitektur spesifik tugas yang menyertakan representasi prapelatihan sebagai fitur tambahan. Pendekatan **fine-tuning**, seperti [*OpenAI GPT*, Radford et al, 2018], memperkenalkan parameter spesifik tugas yang minimal, dan dilatih pada tugas hilir dengan cara menyelaraskan kembali semua parameter prapelatihan [*fine-tuning*]. Kedua pendekatan ini berbagi fungsi objektif yang sama selama prapelatihan, di mana keduanya menggunakan model bahasa **unidirectional** — searah — untuk mempelajari representasi bahasa umum.

Kami berargumen bahwa teknik saat ini membatasi kekuatan representasi prapelatihan, terutama untuk pendekatan **fine-tuning**. Batasan utamanya adalah model bahasa standar bersifat **unidirectional**, dan hal ini membatasi pilihan arsitektur yang dapat digunakan selama prapelatihan. Sebagai contoh, dalam [*OpenAI GPT*, Radford et al, 2018], penulis menggunakan arsitektur kiri-ke-kanan, di mana setiap token hanya dapat memperhatikan (*attend to*) token sebelumnya dalam lapisan **self-attention** dari **Transformer** [*Transformer*, Vaswani et al, 2017]. Pembatasan tersebut bersifat suboptimal untuk tugas tingkat kalimat, dan bisa sangat merugikan saat menerapkan pendekatan berbasis **fine-tuning** pada tugas tingkat token seperti menjawab pertanyaan, di mana penggabungan konteks dari kedua arah sangatlah krusial.

Dalam makalah ini, kami meningkatkan pendekatan berbasis **fine-tuning** dengan mengusulkan **BERT**: **Bidirectional Encoder Representations from Transformers**. **BERT** mengatasi kendala unidireksionalitas yang disebutkan sebelumnya dengan menggunakan objektif prapelatihan **"masked language model"** (MLM), yang terinspirasi oleh tugas **Cloze** [Taylor, 1953]. **Masked language model** secara acak menyamarkan (*masks*) beberapa token dari input, dan tujuannya adalah untuk memprediksi ID kosakata asli dari kata yang disamarkan tersebut hanya berdasarkan konteksnya. Berbeda dengan prapelatihan model bahasa kiri-ke-kanan, objektif MLM memungkinkan representasi untuk menggabungkan konteks kiri dan kanan, yang memungkinkan kami melatih **Transformer** bidireksional yang mendalam. Selain **masked language model**, kami juga menggunakan tugas **"next sentence prediction"** — prediksi kalimat berikutnya — yang melatih representasi pasangan teks secara bersama-sama.

Kontribusi dari makalah kami adalah sebagai berikut:

* Kami mendemonstrasikan pentingnya prapelatihan bidireksional untuk representasi bahasa. Tidak seperti [*OpenAI GPT*, Radford et al, 2018] yang menggunakan model bahasa **unidirectional**, **BERT** menggunakan **masked language model** untuk memungkinkan representasi bidireksional mendalam. Ini juga berbeda dengan [*ELMo*, Peters et al, 2018a], yang menggunakan konkatenasi dangkal dari model bahasa kiri-ke-kanan dan kanan-ke-kiri yang dilatih secara independen.


* Kami menunjukkan bahwa representasi prapelatihan mengurangi kebutuhan akan banyak arsitektur spesifik tugas yang dirancang secara rumit. **BERT** adalah model representasi berbasis **fine-tuning** pertama yang mencapai performa mutakhir (*state-of-the-art*) pada berbagai tugas tingkat kalimat dan tingkat token, melampaui banyak arsitektur spesifik tugas.


* **BERT** memajukan standar pencapaian mutakhir untuk sebelas tugas NLP. Kode dan model prapelatihan tersedia di [https://github.com/google-research/bert](https://github.com/google-research/bert).


## 2 - Related Work

## 2.1 - Unsupervised Feature-based Approaches

## 2.2 - Unsupervised Fine-tuning Approaches

## 2.3 - Transfer Learning from Supervised Data

## 3 - BERT

## 3.1 - Pre-training BERT

## 3.2 - Fine-tuning BERT

## 4 - Experiments

## 4.1 - GLUE

## 4.2 - SQuAD v1.1

## 4.3 - SQuAD v2.0

## 4.4 - SWAG

## 5 - Ablation Studies

## 5.1 - Effect of Pre-training Tasks

## 5.2 - Effect of Model Size

## 5.3 - Feature-based Approach with BERT

## 6 - Conclusion

## A - Additional Details for BERT

## A.1 - Illustration of the Pre-training Tasks

## A.2 - Pre-training Procedure

## A.3 - Fine-tuning Procedure

## A.4 - Comparison of BERT, ELMo, and OpenAI GPT

## A.5 - Illustrations of Fine-tuning on Different Tasks

## B - Detailed Experimental Setup

## B.1 - Detailed Descriptions for the GLUE Benchmark Experiments

## C - Additional Ablation Studies

## C.1 - Effect of Number of Training Steps

## C.2 - Ablation for Different Masking Procedures