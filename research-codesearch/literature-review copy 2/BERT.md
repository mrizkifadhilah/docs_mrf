# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## ABSTRACT

### **Pengenalan Model BERT**
Kami memperkenalkan sebuah model representasi bahasa baru yang diberi nama **BERT**. Nama ini merupakan singkatan dari **Bidirectional Encoder Representations from Transformers**. Secara sederhana, ini adalah sebuah sistem kecerdasan buatan yang dirancang untuk memahami nuansa dan makna bahasa manusia dengan cara menganalisis struktur kalimat.

### **Perbedaan Mendasar dengan Model Sebelumnya**
Pendekatan BERT sangat berbeda jika dibandingkan dengan model-model representasi bahasa terkini lainnya, seperti **[Deep contextualized word representations (ELMo), Peters et al., 2018]** dan **[Improving Language Understanding by Generative Pre-training (OpenAI GPT), Radford et al., 2018]**.

Perbedaannya terletak pada cara model "membaca" teks:
* Model-model sebelumnya umumnya bersifat satu arah (unidirectional). Mereka memahami kata dengan melihat konteks hanya dari kiri ke kanan (atau sebaliknya), mirip seperti kita membaca kalimat dan mencoba menebak kata berikutnya tanpa melihat apa yang ada di depannya.
* Sebaliknya, **BERT dirancang untuk melakukan pra-pelatihan (pre-training) representasi dua arah (bidirectional) yang mendalam.**
* *Penjelasan:* "Pra-pelatihan" berarti model ini belajar pola bahasa secara umum terlebih dahulu dari kumpulan teks yang sangat besar yang belum diberi label (unlabeled text)—seperti membaca seluruh Wikipedia untuk belajar tata bahasa dan fakta dunia tanpa diajari secara spesifik.
* *Konsep Bidirectional:* BERT melakukan "pengkondisian gabungan" (*jointly conditioning*) pada konteks kiri dan kanan di semua lapisan pemrosesannya. Bayangkan Anda membaca sebuah kata sulit di tengah kalimat; untuk memahaminya, Anda pasti melihat kata-kata sebelum **dan** sesudahnya secara bersamaan. Inilah yang dilakukan BERT, sehingga pemahamannya terhadap konteks jauh lebih utuh.

### **Keunggulan: Fleksibilitas Melalui "Fine-Tuning"**
Sebagai hasil dari kemampuan dua arah tersebut, model BERT yang telah melalui tahap pra-pelatihan ini menjadi sangat fleksibel. Model ini dapat melalui proses **"fine-tuning"** (penyesuaian halus untuk tugas spesifik) dengan sangat efisien.
* **Hanya butuh satu lapisan tambahan:** Untuk mengubah BERT dari pembaca umum menjadi ahli dalam tugas tertentu, kita hanya perlu menambahkan satu lapisan keluaran (*output layer*) tambahan.
* **Tanpa modifikasi arsitektur yang rumit:** Kita tidak perlu merombak ulang struktur atau arsitektur model secara besar-besaran untuk setiap tugas yang berbeda.

Ini memungkinkan penciptaan model-model mutakhir (*state-of-the-art*) untuk berbagai macam tugas, mulai dari **sistem tanya-jawab** (*question answering*) hingga **inferensi bahasa** (menentukan hubungan logis antar kalimat).

### **Hasil Eksperimental dan Performa**
Secara konsep, BERT itu sederhana, namun secara empiris (berdasarkan bukti percobaan), ia sangatlah kuat. BERT berhasil memecahkan rekor baru dan mencapai hasil *state-of-the-art* pada **sebelas tugas pemrosesan bahasa alami (NLP)** yang berbeda.

Berikut adalah rincian pencapaian statistik performa BERT:
1. **GLUE Benchmark (General Language Understanding Evaluation):**
* Skor GLUE berhasil didorong hingga mencapai **80.5%**.
* Ini merupakan peningkatan absolut sebesar **7.7% poin** dibandingkan rekor sebelumnya (sebuah lonjakan yang sangat signifikan dalam dunia statistik AI).

2. **MultiNLI (Multi-Genre Natural Language Inference):**
* Tugas ini menguji kemampuan komputer untuk menentukan apakah satu kalimat mendukung, bertentangan, atau netral terhadap kalimat lain.
* Akurasi BERT mencapai **86.7%**.
* Ini mencatatkan peningkatan absolut sebesar **4.6%**.

3. **SQuAD v1.1 (Stanford Question Answering Dataset versi 1.1):**
* Ini adalah ujian kemampuan membaca dan menjawab pertanyaan berdasarkan teks.
* Skor *Test F1* (ukuran ketepatan dan kelengkapan jawaban) mencapai **93.2**.
* Ini merupakan peningkatan absolut sebesar **1.5 poin**.

4. **SQuAD v2.0:**
* Versi ini lebih sulit karena model juga harus mengenali ketika sebuah pertanyaan *tidak memiliki jawaban* di dalam teks.
* Skor *Test F1* mencapai **83.1**.
* Ini mencatatkan peningkatan absolut sebesar **5.1 poin**.

## 1 Introduction
### **Efektivitas Pra-Pelatihan Model Bahasa**
Pra-pelatihan model bahasa (*language model pre-training*) telah terbukti secara ilmiah sebagai metode yang sangat efektif untuk meningkatkan kinerja sistem kecerdasan buatan dalam berbagai tugas pemrosesan bahasa alami (NLP).

Secara sederhana, "pra-pelatihan" ini mirip dengan memberikan pendidikan dasar umum kepada siswa sebelum mereka mempelajari bidang spesifik. Model komputer "membaca" teks dalam jumlah masif terlebih dahulu untuk memahami struktur bahasa, kosa kata, dan konteks umum. Studi-studi yang memvalidasi efektivitas pendekatan ini antara lain:
* [Semi-supervised sequence learning, Dai dan Le, 2015] 
* [Deep contextualized word representations, Peters et al., 2018a] 
* [Improving language understanding with unsupervised learning, Radford et al., 2018] 
* [Universal language model fine-tuning for text classification, Howard dan Ruder, 2018] 

Peningkatan kinerja ini mencakup dua kategori tugas utama, yaitu tugas tingkat kalimat dan tugas tingkat token:

### **1. Tugas Tingkat Kalimat (*Sentence-Level Tasks*)**
Kategori ini mencakup tugas-tugas yang menuntut model untuk menganalisis kalimat secara **holistik** atau menyeluruh. Artinya, model tidak hanya melihat kata per kata, tetapi harus memahami makna utuh dari satu kalimat untuk memprediksi hubungannya dengan kalimat lain. Contoh tugas dalam kategori ini meliputi:
* **Inferensi Bahasa Alami (*Natural Language Inference*):** Ini adalah tugas logika di mana model harus menentukan apakah satu kalimat merupakan kesimpulan yang valid dari kalimat lainnya.
> Referensi studi: [A large annotated corpus for learning natural language inference, Bowman et al., 2015] dan [A broad-coverage challenge corpus for sentence understanding through inference, Williams et al., 2018].

* **Parafrase (*Paraphrasing*):** Kemampuan untuk mengenali atau menghasilkan kalimat berbeda yang memiliki makna yang sama persis.
> Referensi studi: [Automatically constructing a corpus of sentential paraphrases, Dolan dan Brockett, 2005].

### **2. Tugas Tingkat Token (*Token-Level Tasks*)**
Berbeda dengan analisis kalimat utuh, tugas ini menuntut ketelitian yang jauh lebih tinggi. Model diharuskan menghasilkan luaran yang **sangat rinci (*fine-grained*)** pada tingkat "token". Dalam konteks NLP, token biasanya merujuk pada satu kata, bagian dari kata, atau tanda baca. Contoh tugas ini meliputi:

* **Pengenalan Entitas Bernama (*Named Entity Recognition*):** Model harus memindai teks dan menunjuk kata spesifik mana yang merupakan nama orang, lokasi, organisasi, atau entitas lainnya.
> Referensi studi: [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition, Tjong Kim Sang dan De Meulder, 2003].

* **Sistem Tanya Jawab (*Question Answering*):** Model harus membaca sebuah paragraf dan menunjuk deretan kata (token) yang tepat yang berfungsi sebagai jawaban atas pertanyaan yang diajukan.
> Referensi studi: [SQuAD: 100,000+ Questions for Machine Comprehension of Text, Rajpurkar et al., 2016].

### **Dua Strategi Utama Penerapan Model Pra-Pelatihan**
Saat ini, terdapat dua strategi yang sudah ada (*existing strategies*) untuk menerapkan representasi bahasa yang telah dipra-latih ke dalam "tugas hilir" atau *downstream tasks*.
> **Catatan Konsep:** *"Downstream tasks"* adalah istilah teknis untuk tugas spesifik yang kita ingin AI kerjakan setelah ia belajar bahasa secara umum. Ibaratnya, jika pra-pelatihan adalah sekolah dasar (belajar membaca umum), maka *downstream tasks* adalah pekerjaan spesifiknya nanti, seperti menjadi penerjemah, penulis ringkasan, atau analis sentimen.

Kedua strategi tersebut adalah pendekatan **berbasis fitur (*feature-based*)** dan pendekatan **penyesuaian halus (*fine-tuning*)**.

#### **1. Pendekatan Berbasis Fitur (*The Feature-Based Approach*)**

Contoh utama dari pendekatan ini adalah **ELMo** (dijelaskan dalam studi **[Deep contextualized word representations (ELMo), Peters et al., 2018]**).
* **Cara Kerjanya:** Pendekatan ini menggunakan arsitektur khusus yang dirancang spesifik untuk satu tugas tertentu.
* **Peran Model Pra-Pelatihan:** Dalam metode ini, representasi dari model yang sudah dipra-latih dianggap sebagai "fitur tambahan" yang statis.
> **Analogi:** Bayangkan model pra-pelatihan ini sebagai seorang **konsultan ahli**. Konsultan ini memberikan sarannya (fitur/representasi), tetapi Anda tetap harus membangun gedung Anda sendiri (arsitektur khusus tugas) dan menggunakan saran konsultan tersebut hanya sebagai salah satu bahan pertimbangan atau lapisan tambahan, tanpa mengubah si konsultannya itu sendiri.

#### **2. Pendekatan Penyesuaian Halus (*The Fine-Tuning Approach*)**
Contoh dari pendekatan ini adalah **OpenAI GPT** (dijelaskan dalam studi **[Improving Language Understanding by Generative Pre-training (OpenAI GPT), Radford et al., 2018]**).
* **Cara Kerjanya:** Pendekatan ini memperkenalkan jumlah parameter khusus tugas yang sangat minimal.
* **Peran Model Pra-Pelatihan:** Model ini dilatih untuk tugas spesifik (hilir) dengan cara melakukan *fine-tuning* pada **seluruh** parameter yang sudah dipra-latih.
> **Analogi:** Bayangkan model pra-pelatihan ini sebagai seorang **atlet serba bisa**. Untuk membuatnya ahli bermain tenis, Anda tidak menyewa orang lain. Sebaliknya, Anda melatih ulang atlet tersebut secara langsung. Anda mengubah sedikit cara dia bergerak dan berpikir (mengubah seluruh parameter internalnya) agar sesuai dengan aturan tenis. Jadi, model aslinya ikut berubah menyesuaikan tugas baru.

### **Kesamaan dalam Fase Pra-Pelatihan**
Meskipun cara penerapannya berbeda, kedua pendekatan ini (ELMo dan GPT) berbagi fungsi tujuan (*objective function*) yang sama selama masa pra-pelatihan.
> **Penjelasan Konsep:** *"Objective function"* adalah target atau sistem penilaian matematika yang digunakan model saat belajar untuk mengetahui apakah ia sudah benar atau salah.

Keduanya menggunakan **model bahasa satu arah (*unidirectional language models*)** untuk mempelajari representasi bahasa secara umum.
> *Artinya:* Model-model ini belajar membaca teks hanya dari satu arah (misalnya kiri ke kanan saja), yang membatasi kemampuan mereka untuk melihat konteks kalimat secara menyeluruh dibandingkan jika mereka bisa melihat ke dua arah sekaligus.

### **Kritik terhadap Keterbatasan Teknik yang Ada**
Kami mengajukan argumen bahwa teknik-teknik yang digunakan saat ini membatasi kekuatan atau potensi penuh dari representasi bahasa yang telah dipra-latih (*pre-trained representations*). Hambatan ini khususnya terasa sangat signifikan pada pendekatan **penyesuaian halus (*fine-tuning*)**, di mana model dasar yang sudah ada disesuaikan untuk tugas baru.

### **Masalah Utama: Keterbatasan "Satu Arah" (*Unidirectionality*)**
Keterbatasan terbesar yang kami temukan adalah bahwa model bahasa standar saat ini bersifat **satu arah (*unidirectional*)**.
* **Dampak pada Arsitektur:** Sifat satu arah ini sangat membatasi pilihan arsitektur model yang dapat digunakan selama fase pra-pelatihan. Kita menjadi terpaku pada desain yang hanya bisa memproses informasi secara linear, bukan menyeluruh.
* **Contoh Kasus (OpenAI GPT):** Sebagai contoh konkret, dalam model OpenAI GPT, para penulis menggunakan arsitektur **kiri-ke-kanan (*left-to-right*)**.
* Dalam struktur Transformer—sebuah arsitektur jaringan saraf revolusioner yang diperkenalkan dalam **[Attention Is All You Need, Vaswani et al., 2017]**—setiap "token" (kata atau potongan kata) hanya diizinkan untuk memperhatikan atau mengakses informasi dari token-token **sebelumnya** di lapisan *self-attention*.
> *Analogi:* Bayangkan Anda membaca novel misteri, tetapi Anda dipaksa menutup semua kata di sebelah kanan kata yang sedang Anda baca. Anda tidak bisa mengintip ke depan untuk memahami konteks kalimat secara utuh.

### **Dampak Negatif pada Kinerja Tugas**
Pembatasan akses informasi ini memiliki konsekuensi serius terhadap performa model:
1. **Sub-optimal untuk Tugas Tingkat Kalimat:** Pembatasan ini membuat kinerja model menjadi kurang maksimal (*sub-optimal*) untuk tugas-tugas yang membutuhkan pemahaman level kalimat secara keseluruhan.
2. **Sangat Berbahaya untuk Tugas Tingkat Token:** Dampaknya bisa menjadi sangat merugikan (*very harmful*) ketika menerapkan pendekatan *fine-tuning* pada tugas tingkat token, seperti **sistem tanya-jawab (*question answering*)**.
> *Alasan:* Dalam tugas tanya jawab, sangatlah krusial bagi model untuk menggabungkan konteks dari **kedua arah** (kiri dan kanan) secara bersamaan. Untuk menjawab pertanyaan dengan tepat, model harus memahami kata-kata yang muncul baik sebelum maupun sesudah kata kunci yang sedang dianalisis.


### **Solusi BERT: Mengatasi Keterbatasan Satu Arah**
Dalam makalah ini, kami menyempurnakan pendekatan *fine-tuning* (penyesuaian halus) dengan mengajukan model bernama **BERT: Bidirectional Encoder Representations from Transformers**.

Inovasi utama BERT terletak pada kemampuannya memecahkan masalah atau batasan "satu arah" (*unidirectionality constraint*) yang menghambat model-model sebelumnya. Untuk melakukan ini, BERT menggunakan tujuan pra-pelatihan khusus yang disebut **"Masked Language Model" (MLM)**.

### **Konsep "Masked Language Model" (MLM)**
Metode MLM ini terinspirasi oleh tugas **Cloze** yang diperkenalkan dalam studi psikolinguistik klasik **[Cloze procedure: A new tool for measuring readability, Taylor, 1953]**.

Berikut adalah cara kerja mekanismenya:
* **Pengacakan Sinyal (Masking):** Model bahasa bertopeng (MLM) ini secara acak akan menutupi atau "menyembunyikan" sebagian token (kata) dari input teks yang diberikan.
* **Tujuan Prediksi:** Tugas utama model adalah menebak dan memprediksi "id kosakata asli" (*original vocabulary id*) dari kata yang disembunyikan tersebut.
> *Penjelasan:* Model tidak hanya menebak maknanya, tapi harus menemukan kode identitas kata yang tepat dari kamus data yang dimilikinya, dan ia harus melakukan ini hanya dengan mengandalkan konteks kalimat di sekitarnya.

### **Keunggulan MLM Dibandingkan Model Tradisional**
Pendekatan ini sangat berbeda dengan pra-pelatihan model bahasa "kiri-ke-kanan" (*left-to-right*) yang standar.
* Dalam model standar, komputer membaca urut dari awal kalimat, sehingga ia tidak tahu apa yang ada di akhir kalimat saat memproses awal kalimat.
* Sebaliknya, tujuan MLM memungkinkan representasi bahasa untuk **menleburkan (fuse) konteks kiri dan kanan** secara bersamaan.

Kemampuan menggabungkan konteks dari dua arah inilah yang memungkinkan kami untuk melatih sebuah **Transformer dua arah yang mendalam (*deep bidirectional Transformer*)**. Ini ibarat memahami makna sebuah kata di tengah kalimat dengan membaca keseluruhan kalimat tersebut sekaligus, bukan hanya separuh awal.

### **Tugas Tambahan: Prediksi Kalimat Berikutnya**
Selain menggunakan *Masked Language Model*, kami juga menggunakan satu tugas tambahan dalam pelatihan, yaitu **"Next Sentence Prediction"** atau prediksi kalimat berikutnya.
* Tugas ini secara bersamaan melatih representasi untuk **pasangan teks (*text-pair representations*)**.
> *Penjelasan:* Ini melatih model untuk memahami hubungan logis antara dua kalimat yang berbeda (misalnya: "Apakah Kalimat B adalah kelanjutan logis dari Kalimat A?"), yang sangat penting untuk tugas seperti tanya-jawab atau pengambilan kesimpulan.

### **Kontribusi Utama Penelitian Ini**
Dalam makalah ini, kami menguraikan tiga sumbangsih utama kami terhadap perkembangan teknologi pemrosesan bahasa:
#### **1. Membuktikan Pentingnya "Pra-Pelatihan Dua Arah"**
Kami mendemonstrasikan secara nyata betapa krusialnya peran pra-pelatihan dua arah (*bidirectional pre-training*) dalam membangun representasi bahasa. Pendekatan BERT ini memiliki perbedaan fundamental dibandingkan metode-metode sebelumnya:

* **Berbeda dengan OpenAI GPT:** Tidak seperti pendekatan dalam **[Improving Language Understanding by Generative Pre-training (OpenAI GPT), Radford et al., 2018]** yang menggunakan model bahasa satu arah (*unidirectional*—hanya membaca dari kiri ke kanan), BERT menggunakan "model bahasa bertopeng" (*masked language models*). Teknik ini memungkinkan BERT untuk menciptakan representasi dua arah yang **mendalam (*deep bidirectional representations*)**, di mana model memahami konteks dari kiri dan kanan secara simultan di setiap lapisan pemrosesannya.

* **Berbeda dengan ELMo:** Pendekatan kami juga sangat kontras dengan metode dalam **[Deep contextualized word representations (ELMo), Peters et al., 2018]**. ELMo memang menggunakan dua arah, namun metodenya hanyalah "penggabungan dangkal" (*shallow concatenation*). Artinya, ELMo melatih model kiri-ke-kanan dan model kanan-ke-kiri secara terpisah, lalu hasilnya hanya ditempelkan satu sama lain. Sebaliknya, BERT melatih kedua arah tersebut secara bersamaan dalam satu kesatuan sistem yang terintegrasi.

#### **2. Mengurangi Ketergantungan pada Arsitektur Rumit**
Kami menunjukkan bahwa dengan menggunakan representasi yang telah dipra-latih dengan baik, kita dapat mengurangi kebutuhan akan arsitektur spesifik tugas yang dirancang dengan rumit (*heavily-engineered task-specific architectures*).
* Biasanya, untuk setiap tugas bahasa yang berbeda, para insinyur harus merancang struktur model yang sangat spesifik dan kompleks. BERT mengubah paradigma ini.
* BERT adalah model representasi berbasis **penyesuaian halus (*fine-tuning*)** pertama yang berhasil mencapai performa mutakhir (*state-of-the-art*) pada serangkaian besar tugas, baik itu tugas tingkat kalimat maupun tingkat token.
* Hebatnya, BERT mampu mengungguli banyak arsitektur lain yang justru dirancang khusus secara spesifik untuk menangani tugas-tugas tersebut.

#### **3. Memecahkan Rekor Baru di 11 Tugas NLP**
BERT berhasil mendorong batasan teknologi saat ini dengan memajukan status *state-of-the-art* (hasil terbaik yang pernah ada) untuk **sebelas tugas Pemrosesan Bahasa Alami (NLP)** yang berbeda.
* Ini berarti BERT menjadi standar baru untuk akurasi dan kinerja dalam berbagai ujian kecerdasan bahasa komputer.
* Sebagai bentuk kontribusi pada komunitas sains terbuka, seluruh kode pemrograman dan model-model yang telah kami latih tersedia untuk publik dan dapat diakses di: `https://github.com/google-research/bert`.
  
## 2 Related Work
Dalam dunia pengembangan kecerdasan buatan, praktik "pra-pelatihan" (*pre-training*) untuk menciptakan representasi bahasa umum bukanlah hal baru. Terdapat sejarah panjang penelitian di bidang ini, dan pada bagian ini, kami akan meninjau secara singkat pendekatan-pendekatan yang paling banyak digunakan selama ini.

## **2.1 Pendekatan Berbasis Fitur Tanpa Pengawasan (*Unsupervised Feature-based Approaches*)**
Upaya untuk mengajarkan komputer agar dapat memahami dan merepresentasikan kata-kata dalam format yang berlaku secara luas telah menjadi area penelitian yang aktif selama beberapa dekade.
Tujuan utama dari riset ini adalah menciptakan representasi kata yang tidak bergantung pada satu tugas spesifik saja, tetapi bisa diterapkan di berbagai situasi. Penelitian-penelitian ini dapat dibagi menjadi dua era atau kategori metode:

1. **Metode Non-Saraf (*Non-neural Methods*):**
Ini adalah pendekatan statistik tradisional yang digunakan sebelum dominasi *deep learning*. Metode ini berfokus pada analisis frekuensi dan struktur kalimat secara manual atau statistik murni. Beberapa karya kunci dalam kategori ini meliputi:
* **[Class-based n-gram models of natural language, Brown et al., 1992]**.
* **[A framework for learning predictive structures from multiple tasks and unlabeled data, Ando dan Zhang, 2005]**.
* **[Domain adaptation with structural correspondence learning, Blitzer et al., 2006]**.

2. **Metode Saraf (*Neural Methods*):**
Ini adalah pendekatan modern yang menggunakan jaringan saraf tiruan (*neural networks*) untuk memproses bahasa, yang meniru cara kerja neuron di otak manusia secara matematis. Karya kunci dalam kategori ini meliputi:
* **[Distributed representations of words and phrases and their compositionality, Mikolov et al., 2013]**.
* **[Glove: Global vectors for word representation, Pennington et al., 2014]**.

### **Pentingnya "Word Embeddings" (Penyematan Kata)**

Salah satu terobosan terbesar dari metode saraf ini adalah terciptanya apa yang disebut **"Pre-trained Word Embeddings"**.
* **Konsep:** *Word embeddings* mengubah kata-kata menjadi deretan angka (vektor) dalam ruang matematika. Dalam ruang ini, kata-kata yang memiliki makna mirip (seperti "raja" dan "ratu") akan memiliki nilai angka yang berdekatan.
* **Peran dalam Sistem Modern:** Saat ini, *word embeddings* yang telah dilatih sebelumnya merupakan bagian integral dan tak terpisahkan dari sistem Pemrosesan Bahasa Alami (*NLP*) modern.
* **Keunggulan:** Menggunakan *embeddings* yang sudah dipra-latih (sudah belajar dari jutaan teks sebelumnya) menawarkan peningkatan kinerja yang signifikan dibandingkan dengan jika sistem harus mempelajari makna kata dari nol (*from scratch*). Hal ini divalidasi dalam studi **[Word representations: A simple and general method for semi-supervised learning, Turian et al., 2010]**.

### **Teknik Pelatihan Vektor Kata**
Untuk melatih vektor-vektor kata ini agar cerdas, para peneliti menggunakan beberapa tujuan atau target pembelajaran (*objectives*):
1. **Pemodelan Bahasa Kiri-ke-Kanan (*Left-to-Right Language Modeling*):**
Dalam metode ini, model dilatih untuk memprediksi kata berikutnya dalam sebuah kalimat berdasarkan kata-kata sebelumnya.
> * *Analogi:* Seperti melengkapi kalimat "Ayah sedang membaca ko..." dengan menebak kata "koran".
>> * Referensi: **[A scalable hierarchical distributed language model, Mnih dan Hinton, 2009]**.

2. **Diskriminasi Konteks Kiri dan Kanan:**
Selain prediksi urut, ada juga metode yang melatih model untuk membedakan antara kata yang "benar" dan kata yang "salah" atau tidak nyambung jika diletakkan di tengah konteks kiri dan kanan tertentu.
> *Analogi:* Model diberikan kalimat "Ibu memasak *batu* di dapur" dan harus menyadari bahwa kata "batu" adalah penyusup yang salah dalam konteks tersebut.
>> Referensi: **[Distributed representations of words and phrases and their compositionality, Mikolov et al., 2013]**.

### **Perluasan ke Unit Bahasa yang Lebih Besar**
Pendekatan-pendekatan pembelajaran representasi bahasa yang sebelumnya berfokus pada kata (seperti *word embeddings*), kini telah digeneralisasi atau diperluas ke **granularitas yang lebih kasar (*coarser granularities*)**.
Dalam konteks ini, "granularitas yang lebih kasar" berarti kita tidak lagi hanya melatih komputer untuk memahami satu per satu kata secara terisolasi, melainkan melatihnya untuk memahami unit bahasa yang lebih besar dan kompleks sebagai satu kesatuan utuh. Perluasan ini mencakup:
* **Penyematan Kalimat (*Sentence Embeddings*):** Komputer belajar merepresentasikan makna satu kalimat penuh ke dalam bentuk vektor matematika.
> Studi terkait: [Skip-thought vectors, Kiros et al., 2015] dan [An efficient framework for learning sentence representations, Logeswaran dan Lee, 2018].
* **Penyematan Paragraf (*Paragraph Embeddings*):** Komputer belajar memahami makna dari sekumpulan kalimat yang membentuk paragraf.
> Studi terkait: [Distributed representations of sentences and documents, Le dan Mikolov, 2014].

### **Metode Melatih Representasi Kalimat**
Untuk melatih komputer agar mampu menghasilkan representasi kalimat yang akurat, penelitian-penelitian terdahulu (*prior work*) menggunakan berbagai tujuan pelatihan (*objectives*). Tujuan pelatihan ini adalah "tugas" yang diberikan kepada komputer agar ia belajar pola bahasa. Terdapat tiga metode utama yang digunakan:
1. **Memeringkat Kandidat Kalimat Berikutnya (*Ranking Candidate Next Sentences*)**
Dalam metode ini, komputer diberikan sebuah kalimat, lalu diminta untuk memilih kalimat mana yang paling masuk akal untuk menjadi kelanjutannya dari sekumpulan pilihan (kandidat).
> *Analogi:* Seperti soal pilihan ganda "Lengkapi percakapan berikut," di mana komputer harus membedakan mana respons yang nyambung dan mana yang tidak.
>> Studi terkait: [Discourse-based objectives for fast unsupervised sentence representation learning, Jernite et al., 2017] dan [An efficient framework for learning sentence representations, Logeswaran dan Lee, 2018].

2. **Pembangkitan Kata Kiri-ke-Kanan (*Left-to-Right Generation*)**
Metode ini menuntut komputer untuk menghasilkan (menulis ulang) kata-kata dari kalimat berikutnya secara urut dari kiri ke kanan, hanya dengan bermodalkan representasi dari kalimat sebelumnya.
> *Analogi:* Jika saya memberi tahu Anda "Hari ini hujan deras," Anda harus bisa menebak dan menulis kalimat selanjutnya kata demi kata, misalnya "Maka... saya... bawa... payung."
>> Studi terkait: [Skip-thought vectors, Kiros et al., 2015].

3. **Tujuan Turunan "Denoising Auto-encoder"**
Metode ini menggunakan teknik *auto-encoder* penghilang derau (*denoising*).
> *Penjelasan:* Komputer diberikan kalimat yang sudah dirusak (misalnya ada kata yang hilang atau urutannya diacak), lalu ditugaskan untuk memulihkan kalimat tersebut ke bentuk aslinya yang benar. Ini melatih komputer untuk memahami struktur dan makna kalimat yang seharusnya.
>> Studi terkait: [Learning distributed representations of sentences from unlabelled data, Hill et al., 2016].

### **Evolusi Representasi Kontekstual: ELMo**
ELMo (*Embeddings from Language Models*) beserta pendahulunya menggeneralisasi atau memperluas penelitian "penyematan kata" (*word embedding*) tradisional ke dimensi yang berbeda.
* **Pergeseran Paradigma:** Jika metode tradisional memperlakukan satu kata sebagai satu kode statis (misalnya, kata "bisa" selalu memiliki kode yang sama entah itu berarti "racun" atau "mampu"), ELMo memperkenalkan pendekatan yang lebih dinamis.
* **Referensi:** Konsep ini dikembangkan dalam studi **[Semi-supervised sequence tagging with bidirectional language models, Peters et al., 2017]** dan **[Deep contextualized word representations, Peters et al., 2018a]**.

#### **Mekanisme Kerja ELMo**
Cara ELMo bekerja adalah dengan mengekstrak fitur-fitur yang **sensitif terhadap konteks (*context-sensitive features*)**. Ia melakukan ini dengan menggunakan dua model bahasa yang terpisah:
1. **Model Kiri-ke-Kanan (*Left-to-Right*):** Membaca kalimat dari awal ke akhir.
2. **Model Kanan-ke-Kiri (*Right-to-Left*):** Membaca kalimat dari akhir ke awal (mundur).

Representasi kontekstual untuk setiap "token" (kata) kemudian dibuat melalui proses **penggabungan (*concatenation*)**.
* *Penjelasan:* ELMo mengambil hasil pembacaan dari model kiri-ke-kanan dan hasil dari model kanan-ke-kiri, lalu "menempelkan" keduanya menjadi satu kesatuan data. Ini berbeda dengan model yang benar-benar dua arah (*deeply bidirectional*) di mana kedua arah berinteraksi di setiap lapisan; ELMo memprosesnya secara terpisah lalu menggabungkannya di akhir.

#### **Dampak pada Kinerja Tugas NLP**
Ketika penyematan kata kontekstual dari ELMo ini diintegrasikan ke dalam arsitektur spesifik tugas yang sudah ada, ELMo terbukti memajukan status *state-of-the-art* (memecahkan rekor kinerja terbaik) untuk beberapa tolak ukur (*benchmarks*) utama dalam Pemrosesan Bahasa Alami (NLP). Tugas-tugas tersebut meliputi:
* **Sistem Tanya Jawab (*Question Answering*):** Kemampuan komputer menjawab pertanyaan berdasarkan teks bacaan.
> Referensi: **[Squad: 100,000+ questions for machine comprehension of text, Rajpurkar et al., 2016]**.

* **Analisis Sentimen (*Sentiment Analysis*):** Kemampuan mendeteksi emosi atau opini (positif/negatif) dalam teks.
> Referensi: **[Recursive deep models for semantic compositionality over a sentiment treebank, Socher et al., 2013]**.

* **Pengenalan Entitas Bernama (*Named Entity Recognition*):** Kemampuan mengidentifikasi nama orang, organisasi, atau lokasi dalam teks.
> Referensi: **[Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition, Tjong Kim Sang dan De Meulder, 2003]**.

### **Pendekatan Serupa Lainnya**
Selain ELMo, terdapat penelitian lain yang mencoba pendekatan serupa namun memiliki karakteristik yang membedakannya dari model yang sepenuhnya dua arah (*deeply bidirectional*):
1. **Melamud et al. (2016):**
* Mereka mengusulkan pembelajaran representasi kontekstual melalui tugas memprediksi satu kata tunggal dengan melihat konteks kiri dan kanannya sekaligus menggunakan **LSTMs (*Long Short-Term Memory*)**.
> *Catatan:* LSTM adalah jenis jaringan saraf tiruan yang mampu mengingat pola jangka panjang dalam urutan data.

* Mirip dengan ELMo, model mereka bersifat **berbasis fitur (*feature-based*)** dan **tidak mendalam secara dua arah (*not deeply bidirectional*)**. Artinya, meskipun melihat kiri dan kanan, interaksi antar arahnya tidak terjadi di setiap lapisan secara mendalam.
> Referensi: **[context2vec: Learning generic context embedding with bidirectional LSTM, Melamud et al., 2016]**.

2. **Fedus et al. (2018):**
* Penelitian ini menunjukkan bahwa **tugas "cloze"** (tugas mengisi bagian kosong dalam kalimat) dapat digunakan untuk meningkatkan ketangguhan (*robustness*) dari model pembangkitan teks (*text generation models*).
> Referensi: **[Maskgan: Better text generation via filling in the..., Fedus et al., 2018]**.

## 2.2 Unsupervised Fine-tunning Approaches
Sama halnya dengan pendekatan berbasis fitur (*feature-based approaches*) yang telah dibahas sebelumnya, penelitian-penelitian awal dalam arah ini dimulai dengan langkah yang sederhana. Awalnya, fokus utama hanyalah melakukan pra-pelatihan pada parameter **penyematan kata (*word embedding*)** saja, menggunakan teks yang tidak berlabel.
* **Penjelasan:** Ini berarti pada masa awal, komputer hanya diajari definisi kata per kata dari kamus data raksasa, tanpa diajari struktur kalimat yang kompleks.
* **Referensi:** [A unified architecture for natural language processing: Deep neural networks with multitask learning, Collobert dan Weston, 2008].

### **Evolusi ke Representasi Kontekstual**
Namun, baru-baru ini terjadi perkembangan signifikan. Para peneliti mulai menggunakan **pengode kalimat atau dokumen (*sentence or document encoders*)** yang lebih canggih.
Berbeda dengan model lama yang hanya melihat kata secara kaku, pengode (encoder) ini menghasilkan **representasi token kontekstual**.
* *Konsep:* Komputer tidak lagi melihat kata "bank" hanya sebagai satu kode statis, tetapi bisa membedakan "bank" (institusi keuangan) dan "bank" (tepi sungai) berdasarkan kalimat sekitarnya.

Prosesnya berjalan dalam dua tahap:
1. **Pra-pelatihan (*Pre-training*):** Model dilatih menggunakan teks tanpa label (seperti membaca seluruh artikel internet tanpa kunci jawaban) untuk memahami pola bahasa umum.
2. **Penyesuaian Halus (*Fine-tuning*):** Model yang sudah "pintar" secara umum tersebut kemudian dilatih ulang secara spesifik untuk **tugas hilir yang diawasi (*supervised downstream task*)**. Ini ibarat seorang sarjana umum yang kemudian mengambil kursus singkat spesifik untuk menjadi ahli hukum atau dokter.

Referensi untuk metode ini meliputi:
* [Semi-supervised sequence learning, Dai dan Le, 2015].
* [Universal language model fine-tuning for text classification, Howard dan Ruder, 2018].
* [Improving language understanding with unsupervised learning, Radford et al., 2018].

### **Keunggulan Efisiensi Parameter**

Keuntungan utama dari pendekatan-pendekatan ini adalah efisiensinya. Hanya **sedikit parameter** yang perlu dipelajari dari nol (*from scratch*) saat model menghadapi tugas baru.
* *Analogi:* Karena model sudah memiliki "pengetahuan dasar" yang luas dari pra-pelatihan, kita tidak perlu mengajarinya mengeja atau tata bahasa lagi saat ingin mengajarinya melakukan analisis sentimen. Kita hanya perlu menyesuaikan sedikit bagian dari otaknya.

### **Bukti Keberhasilan: OpenAI GPT**
Setidaknya sebagian karena keuntungan efisiensi inilah, model **OpenAI GPT** berhasil mencapai hasil *state-of-the-art* (rekor kinerja terbaik) sebelumnya pada banyak tugas tingkat kalimat dari tolak ukur **GLUE (*General Language Understanding Evaluation*)**.

* **Referensi:**
* [Improving language understanding with unsupervised learning, Radford et al., 2018].
* [GLUE: A multi-task benchmark and analysis platform for natural language understanding, Wang et al., 2018a].

### **Teknik Pelatihan yang Digunakan**
Untuk melakukan pra-pelatihan pada model-model canggih tersebut, para peneliti menggunakan beberapa tujuan teknis (*objectives*), antara lain:
1. **Pemodelan Bahasa Kiri-ke-Kanan (*Left-to-Right Language Modeling*):** Melatih model menebak kata berikutnya dalam urutan maju.
2. **Tujuan Auto-encoder:** Melatih model untuk memadatkan informasi (mengompresi) lalu menyusunnya kembali (rekonstruksi), yang memaksa model memahami inti sari data.

Referensi penggunaan teknik ini terdapat dalam:
* [Universal language model fine-tuning for text classification, Howard dan Ruder, 2018].
* [Improving language understanding with unsupervised learning, Radford et al., 2018].
* [Semi-supervised sequence learning, Dai dan Le, 2015].
  
## 2.3 Transfer Learning from Supervised Data
Selain metode tanpa pengawasan (*unsupervised*) yang menggunakan teks mentah, terdapat juga penelitian yang menunjukkan efektivitas dari **transfer pengetahuan** yang berasal dari tugas-tugas yang **terawasi (*supervised tasks*)** dengan kumpulan data yang sangat besar.
> **Konsep "Terawasi":** Dalam konteks AI, "terawasi" berarti model belajar dari data yang sudah memiliki label atau "kunci jawaban" yang jelas (misalnya, kalimat A adalah terjemahan benar dari kalimat B).

Keberhasilan transfer pengetahuan dari data terawasi ini terlihat pada dua bidang utama dalam pemrosesan bahasa:
> **Inferensi Bahasa Alami (*Natural Language Inference*):** Komputer belajar logika bahasa dari pasangan kalimat yang sudah diberi label hubungan logikanya. Hal ini ditunjukkan dalam studi **[Supervised learning of universal sentence representations from natural language inference data, Conneau et al., 2017]**.
>> **Penerjemahan Mesin (*Machine Translation*):** Komputer belajar struktur bahasa dengan mempelajari jutaan pasang kalimat terjemahan antar bahasa. Hal ini dibahas dalam studi **[Learned in translation: Contextualized word vectors, McCann et al., 2017]**.

### **Paralel dengan Dunia Visi Komputer**
Pentingnya strategi ini sebenarnya bercermin pada kesuksesan di bidang lain, yaitu penelitian **Visi Komputer (*Computer Vision*)**—bidang ilmu yang melatih komputer untuk "melihat" dan mengenali gambar.

Riset di bidang ini telah mendemonstrasikan betapa krusialnya peran pembelajaran transfer dari model-model besar yang telah dipra-latih. Sebuah "resep" atau metode yang terbukti sangat efektif di dunia visi komputer adalah:
1. Melakukan pra-pelatihan model menggunakan **ImageNet** (sebuah basis data gambar raksasa yang menjadi standar global untuk pengenalan objek visual).
2. Kemudian melakukan *fine-tuning* pada model tersebut untuk tugas spesifik yang diinginkan.

Pendekatan ini divalidasi oleh penelitian-penelitian berikut:
* **[ImageNet: A Large-Scale Hierarchical Image Database, Deng et al., 2009]**.
* **[How transferable are features in deep neural networks?, Yosinski et al., 2014]**.

## 3 BERT
Pada bagian ini, kami memperkenalkan BERT secara mendalam beserta detail implementasinya. Kerangka kerja (*framework*) kami terdiri dari dua tahapan utama yang berurutan, yaitu **pra-pelatihan (*pre-training*)** dan **penyesuaian halus (*fine-tuning*)**.

### **Tahap 1: Pra-Pelatihan (*Pre-training*)**
Selama fase pra-pelatihan, model dilatih menggunakan **data tak berlabel (*unlabeled data*)** melalui berbagai tugas pra-pelatihan yang berbeda.
> *Penjelasan Konsep:* "Data tak berlabel" berarti teks mentah (seperti isi buku atau artikel Wikipedia) yang tidak memiliki kunci jawaban atau kategori khusus. Di sini, model belajar memahami struktur bahasa, tata bahasa, dan fakta dunia secara mandiri tanpa didikte, mirip dengan seseorang yang membaca seluruh perpustakaan untuk mendapatkan pengetahuan umum yang luas.

### **Tahap 2: Penyesuaian Halus (*Fine-tuning*)**
Untuk fase penyesuaian halus, prosesnya dimulai dengan menginisialisasi model BERT menggunakan parameter-parameter yang sudah dipelajari dari tahap pra-pelatihan.
> *Penjelasan:* "Parameter" adalah jutaan angka internal (bobot) dalam otak digital model yang merepresentasikan pengetahuannya. "Inisialisasi" berarti kita tidak memulai dari nol (otak kosong), melainkan memulai dengan otak yang sudah "berpendidikan" dari tahap pertama.

Selanjutnya, **seluruh** parameter tersebut disesuaikan kembali (*fine-tuned*) menggunakan **data berlabel (*labeled data*)** yang berasal dari tugas-tugas hilir (*downstream tasks*).
> *Penjelasan:* "Data berlabel" adalah soal latihan yang memiliki kunci jawaban yang benar (misalnya, pasangan pertanyaan dan jawaban yang tepat). Ini adalah tahap spesialisasi.

### **Model Terpisah untuk Setiap Tugas**
Penting untuk dicatat bahwa setiap tugas hilir akan memiliki model hasil *fine-tuned* yang terpisah, meskipun semuanya berawal dari satu sumber yang sama, yaitu parameter pra-pelatihan yang identik.
> *Analogi:* Bayangkan parameter pra-pelatihan sebagai "pendidikan sarjana umum". Dari satu lulusan sarjana ini, kita bisa mencetak satu model menjadi "Dokter" (untuk tugas medis), satu menjadi "Hakim" (untuk tugas hukum), dan satu menjadi "Penerjemah". Mereka berasal dari basis otak yang sama, tetapi hasil akhirnya adalah spesialis yang berbeda-beda.

Sebagai panduan visual sepanjang bagian ini, kami akan menggunakan contoh kasus **sistem tanya-jawab (*question-answering*)** seperti yang diilustrasikan pada **Gambar 1**.

### **Arsitektur yang Terpadu (*Unified Architecture*)**
Salah satu fitur yang paling membedakan (distingtif) dari BERT adalah **arsitektur terpadu (*unified architecture*)** yang digunakannya di berbagai tugas yang berbeda.

Terdapat perbedaan yang sangat **minimal** antara arsitektur yang digunakan saat pra-pelatihan dengan arsitektur akhir yang digunakan untuk tugas hilir (*downstream*).
> *Implikasi:* Hal ini berbeda dengan sistem lain yang sering kali membutuhkan penambahan komponen rumit atau struktur tambahan yang berat untuk tugas tertentu. Dengan BERT, struktur "otak" digital yang dipakai saat belajar membaca umum (pra-pelatihan) hampir sama persis dengan struktur yang dipakai saat bekerja spesifik (penyesuaian halus).

### **Arsitektur Model BERT**
Arsitektur model yang digunakan oleh BERT adalah sebuah **enkoder Transformer dua arah yang berlapis-lapis (*multi-layer bidirectional Transformer encoder*)**.

Untuk memahami istilah teknis ini, mari kita bedah satu per satu:
* **Transformer:** Ini adalah jenis arsitektur jaringan saraf tiruan modern yang sangat efisien dalam menangani data urutan (seperti kalimat), karena ia menggunakan mekanisme "atensi" (*attention*) untuk menimbang pentingnya setiap kata terhadap kata lainnya.
* **Encoder:** Bagian dari Transformer yang bertugas "membaca" dan mengolah input teks menjadi representasi numerik yang kaya makna. BERT hanya menggunakan bagian pembaca ini, bukan bagian penghasil teks (*decoder*).
* **Multi-layer (Berlapis-lapis):** BERT tidak hanya memproses data satu kali, tetapi melewatkannya melalui tumpukan lapisan pemrosesan yang berulang-ulang untuk menangkap nuansa bahasa yang semakin dalam di setiap lapisannya.
* **Bidirectional (Dua Arah):** Seperti yang dijelaskan sebelumnya, model ini melihat konteks dari kiri dan kanan sekaligus.

#### **Dasar Implementasi**
Desain ini didasarkan pada implementasi asli yang dijelaskan dalam makalah terobosan **[Attention is all you need, Vaswani et al., 2017]**. Implementasi kodenya sendiri dirilis dalam pustaka perangkat lunak `tensor2tensor`.

#### **Mengapa Tidak Dijelaskan Secara Rinci?**
Dalam makalah ini, kami memutuskan untuk **tidak** memberikan deskripsi latar belakang yang mendetail atau melelahkan mengenai arsitektur model ini. Alasannya adalah:
1. Penggunaan Transformer telah menjadi sangat umum (*become common*) dan menjadi standar industri dalam bidang ini.
2. Implementasi yang kami gunakan hampir **identik** (*almost identical*) dengan versi aslinya.

#### **Rekomendasi Bacaan Lanjutan**
Oleh karena itu, bagi pembaca yang ingin memahami detail teknis mendalam mengenai cara kerja Transformer, kami menyarankan untuk merujuk pada:
> Makalah asli: **[Attention is all you need, Vaswani et al., 2017]**.
>> Panduan-panduan berkualitas tinggi lainnya, seperti artikel daring berjudul **“The Annotated Transformer”** (dapat diakses di: `http://nlp.seas.harvard.edu/2018/04/03/attention.html`).

### **Spesifikasi Teknis dan Dimensi Model**
Dalam penelitian ini, kami menetapkan notasi standar untuk menggambarkan struktur arsitektur (kerangka) model kami:
* **L (Layers):** Melambangkan jumlah lapisan atau tumpukan blok Transformer. Bayangkan ini sebagai "kedalaman" pemikiran model; semakin banyak lapisannya, semakin kompleks pola yang bisa diproses.
* **H (Hidden Size):** Melambangkan ukuran tersembunyi. Ini adalah kapasitas penyimpanan informasi untuk setiap kata (token). Semakin besar angka ini, semakin kaya nuansa makna yang bisa direpresentasikan.
* **A (Attention Heads):** Melambangkan jumlah "kepala atensi mandiri". Ini ibarat jumlah "lensa" atau sudut pandang berbeda yang digunakan model untuk menganalisis hubungan antar kata secara bersamaan.

Dalam semua kasus, kami mengatur ukuran lapisan *feed-forward* (filter pemroses internal) sebesar **4H**, atau empat kali lipat dari ukuran tersembunyi.
* Secara spesifik, ini berarti ukurannya adalah **3072** untuk model dengan H = 768.
* Dan ukurannya adalah **4096** untuk model dengan H = 1024.

### **Varian Model: Base vs. Large**
Kami menyajikan hasil utama penelitian berdasarkan dua ukuran model yang berbeda:

1. **BERT BASis ()**
* Jumlah Lapisan () = 12
* Ukuran Tersembunyi () = 768
* Kepala Atensi () = 12
* **Total Parameter = 110 Juta**. (*Catatan: Parameter adalah variabel atau "bobot" internal yang dipelajari model selama latihan, mirip dengan sinapsis di otak.*) 

2. **BERT BESAR ()**
* Jumlah Lapisan () = 24
* Ukuran Tersembunyi () = 1024
* Kepala Atensi () = 16
* **Total Parameter = 340 Juta**.

### **Perbandingan Strategis dengan OpenAI GPT**
Spesifikasi **** dipilih secara sengaja agar memiliki ukuran model yang setara dengan **OpenAI GPT**. Tujuannya adalah untuk keperluan perbandingan yang adil; kami ingin memastikan bahwa jika BERT lebih unggul, itu karena desain arsitekturnya yang lebih pintar, bukan hanya karena ukurannya yang lebih besar.

**Perbedaan Kritis: Mekanisme Atensi**
Meskipun ukurannya sama, terdapat perbedaan arsitektur yang sangat kritis dan mendasar:

* **BERT:** Menggunakan **atensi mandiri dua arah (*bidirectional self-attention*)**. Artinya, saat memproses satu kata, BERT bisa "melihat" kata-kata sebelum dan sesudahnya sekaligus.
* **GPT:** Menggunakan **atensi mandiri terbatas (*constrained self-attention*)**. Dalam model ini, setiap token (kata) hanya diizinkan untuk memperhatikan konteks di sebelah **kirinya**. Model ini "buta" terhadap kata-kata yang muncul setelahnya dalam urutan.

### **Catatan Terminologi Literatur**
Penting untuk dicatat bahwa dalam literatur ilmiah, kedua jenis arsitektur ini memiliki istilah teknis yang berbeda:
* Transformer dua arah (seperti BERT) sering disebut sebagai **"Transformer encoder"** (pengode) karena fungsinya yang kuat dalam memahami dan mengodekan makna input secara utuh.
* Transformer versi konteks-kiri-saja (seperti GPT) sering disebut sebagai **"Transformer decoder"** (pendekode).
> *Alasan:* Sifatnya yang membaca satu arah membuatnya ideal untuk tugas **pembangkitan teks (*text generation*)**, seperti menulis cerita di mana kata berikutnya bergantung pada kata-kata yang sudah ditulis sebelumnya.


### **Representasi Input/Output**

Agar BERT mampu menangani berbagai macam tugas hilir (*down-stream tasks*)—seperti klasifikasi teks atau tanya jawab—kami merancang representasi input yang sangat fleksibel. Representasi ini mampu menggambarkan dua jenis struktur data secara tegas dan tanpa ambiguitas dalam satu **urutan token (*token sequence*)**:

1. **Kalimat Tunggal:** Input yang hanya terdiri dari satu potong teks. 


2. **Pasangan Kalimat:** Input yang terdiri dari dua potong teks yang berpasangan (misalnya: format ⟨Pertanyaan, Jawaban⟩). 



Hal ini memungkinkan BERT memproses berbagai skenario komunikasi hanya dalam satu baris kode input.

### **Definisi Teknis: "Kalimat" vs. "Urutan"**

Untuk menghindari kebingungan, kami perlu mendefinisikan ulang beberapa istilah linguistik umum yang digunakan secara khusus dalam penelitian ini:

* **Definisi "Kalimat" (*Sentence*):**
Di sepanjang karya tulis ini, istilah "kalimat" dapat merujuk pada rentang teks yang bersambung (*contiguous span of text*) secara sembarang. 
> *Penjelasan:* Ini berbeda dengan definisi "kalimat linguistik" yang sebenarnya (yang harus memiliki struktur tata bahasa lengkap seperti subjek, predikat, dan diakhiri tanda titik).  Bagi BERT, sebuah "kalimat" bisa saja berupa satu paragraf panjang, atau hanya berupa potongan frasa pendek, asalkan teksnya bersambung.

* **Definisi "Urutan" (*Sequence*):**
Istilah "urutan" merujuk pada rangkaian token input yang sebenarnya dimasukkan ke dalam mesin BERT untuk diproses. 
>*Penjelasan:* Sebuah "urutan" ini bertindak sebagai wadah atau paket data. Wadah ini bisa berisi **satu kalimat tunggal** saja, atau bisa juga berisi **dua kalimat** yang dikemas (*packed*) bersama-sama menjadi satu kesatuan input.

Berikut adalah penjelasan mendalam mengenai teknis representasi input dalam BERT, diuraikan agar mudah dipahami:

### **Sistem Kosakata: WordPiece Embeddings**
Untuk memproses bahasa, kami menggunakan sistem penyematan (*embeddings*) yang disebut **WordPiece**.
* **Ukuran Kosakata:** Kami menggunakan daftar kosakata sebanyak **30.000 token**.
* **Konsep WordPiece:** Alih-alih menyimpan setiap kata sebagai satu unit utuh (yang membutuhkan kamus jutaan kata), WordPiece memecah kata menjadi potongan-potongan sub-kata yang umum.
> *Analogi:* Bayangkan bermain Lego. Daripada memiliki satu balok khusus untuk setiap bentuk rumah, mobil, atau pesawat, kita memiliki balok-balok dasar (sub-kata) yang bisa disusun menjadi bentuk apa saja. Ini memungkinkan model memahami kata-kata langka atau kompleks dengan melihat potongan penyusunnya.
>> Referensi teknis: **[Google’s neural machine translation system: Bridging the gap between human and machine translation, Wu et al., 2016]**.



### **Token Spesial dan Struktur Urutan**
Dalam setiap urutan input yang dimasukkan ke BERT, kami menyisipkan token-token spesial yang berfungsi sebagai rambu lalu lintas bagi model:
1. **Token Klasifikasi ([CLS]):**
* Token pertama dari setiap urutan **selalu** berupa token klasifikasi khusus yang diberi label `[CLS]`.
> *Fungsi:* Token ini bertindak sebagai "pengumpul informasi". Keadaan tersembunyi akhir (*final hidden state*) yang berhubungan dengan token ini digunakan sebagai **representasi urutan agregat**.
>> *Penjelasan:* Saat model selesai membaca kalimat, seluruh pemahaman inti tentang kalimat tersebut ("apakah ini kalimat positif?", "apakah ini spam?") akan dipadatkan dan disimpan di dalam token `[CLS]` ini untuk keperluan tugas klasifikasi.


2. **Pengemasan Pasangan Kalimat:**
* Seperti dijelaskan sebelumnya, kami sering memasukkan pasangan kalimat (misalnya Kalimat A dan Kalimat B) ke dalam satu urutan tunggal (*single sequence*).

### **Membedakan Kalimat dalam Satu Urutan**
Karena dua kalimat digabung menjadi satu baris panjang, kami membedakannya dengan dua cara agar model tidak bingung:

1. **Pemisah Token ([SEP]):**
* Kami memisahkan kedua kalimat tersebut dengan menyisipkan token spesial bernama `[SEP]` di antara keduanya. Ini berfungsi sebagai tembok pembatas yang jelas.




2. **Penyematan Segmen (*Segment Embeddings*):**
* Kami menambahkan "penyematan yang dipelajari" (*learned embedding*) ke setiap token.
> *Penjelasan:* Setiap kata diberikan "tanda pengenal digital" tambahan yang memberi tahu model: "Saya adalah bagian dari Kalimat A" atau "Saya adalah bagian dari Kalimat B". Ini membantu model memahami struktur wacana.

### **Notasi Matematika dan Vektor**
Untuk keperluan formalisasi dalam rumus matematika, kami menggunakan notasi berikut sebagaimana ditunjukkan dalam ilustrasi arsitektur (Gambar 1):

* ****: Melambangkan **penyematan input** (*input embedding*), yaitu bentuk awal data saat masuk ke model.
* ****: Melambangkan **vektor tersembunyi akhir** untuk token spesial `[CLS]`.
> *Penjelasan:* Simbol  berarti vektor ini adalah deretan angka riil yang memiliki dimensi sebesar  (ukuran tersembunyi model). Vektor  inilah yang dipakai untuk klasifikasi.

* ****: Melambangkan **vektor tersembunyi akhir** untuk token input ke-.
> *Penjelasan:* Ini adalah representasi pemahaman model terhadap kata ke- setelah melalui seluruh proses pengolahan.

Berikut adalah penjelasan detail mengenai bagaimana BERT membangun representasi input untuk setiap kata, yang diuraikan agar mudah dipahami secara visual dan konseptual:
### **Konstruksi Representasi Input**

Untuk setiap "token" (kata atau potongan kata) yang diproses, representasi inputnya tidak berdiri sendiri. Representasi ini dibangun dengan cara **menjumlahkan (*summing*)** tiga lapisan penyematan (*embedding*) yang berbeda secara matematis.

Artinya, satu kata yang masuk ke dalam otak BERT sebenarnya membawa tiga "kartu identitas" sekaligus yang ditumpuk menjadi satu. Ketiga lapisan tersebut adalah:
1. **Penyematan Token (*Token Embeddings*):** Ini adalah identitas dasar kata tersebut. Misalnya, kode unik untuk kata "anjing" atau "lari".
2. **Penyematan Segmen (*Segment Embeddings*):** Ini adalah penanda lokasi kalimat. Kode ini memberi tahu model apakah kata tersebut berasal dari Kalimat A (kalimat pertama) atau Kalimat B (kalimat kedua).
3. **Penyematan Posisi (*Position Embeddings*):** Ini adalah penanda urutan. Karena model matematika dasar tidak mengerti urutan waktu, kode ini memberi tahu model bahwa "ini adalah kata ke-1", "ini kata ke-5", dan seterusnya.

Visualisasi dari proses konstruksi ini—bagaimana ketiga lapisan tersebut ditumpuk dan dijumlahkan—dapat dilihat pada ilustrasi berikut (merujuk pada Gambar 2 dalam dokumen asli).

## 3.1 Pre-training BERT

Pendekatan kami dalam melatih BERT memiliki perbedaan fundamental jika dibandingkan dengan model-model pendahulunya, seperti yang tercantum dalam studi **[Deep contextualized word representations (ELMo), Peters et al., 2018]** dan **[Improving Language Understanding by Generative Pre-training (OpenAI GPT), Radford et al., 2018]**.

Kami **tidak** menggunakan model bahasa tradisional yang bekerja secara satu arah, baik itu model kiri-ke-kanan (*left-to-right*) maupun kanan-ke-kiri (*right-to-left*).
> *Penjelasan:* Model tradisional bekerja seperti manusia membaca: memprediksi kata berikutnya berdasarkan kata-kata sebelumnya. Namun, metode ini membatasi pemahaman konteks secara menyeluruh.

Sebagai gantinya, kami melakukan pra-pelatihan pada BERT dengan menggunakan **dua tugas tanpa pengawasan (*unsupervised tasks*)** yang berbeda.
> *Konsep "Tanpa Pengawasan":* Ini berarti model belajar dari data teks mentah (seperti artikel atau buku) tanpa memerlukan label atau kunci jawaban yang dibuat oleh manusia. Model belajar menemukan pola sendiri dari struktur data tersebut.

Detail mengenai kedua tugas ini akan dijelaskan pada bagian selanjutnya. Tahapan proses ini juga dapat dilihat visualisasinya pada bagian kiri dari **Gambar 1**.

### **Tugas #1: Model Bahasa Bertopeng (*Masked LM*)**

Secara intuitif, sangat masuk akal untuk meyakini bahwa sebuah **model dua arah yang mendalam (*deep bidirectional model*)** memiliki kekuatan yang secara mutlak lebih unggul (*strictly more powerful*) dibandingkan dengan dua pendekatan tradisional lainnya:

1. **Model Kiri-ke-Kanan:** Model yang hanya membaca dari awal ke akhir.
2. **Penggabungan Dangkal (*Shallow Concatenation*):** Metode yang menggabungkan hasil dari model kiri-ke-kanan dan model kanan-ke-kiri yang dilatih secara terpisah (seperti pada ELMo).

**Mengapa "Mendalam" itu Lebih Baik?**
Dalam model "mendalam" seperti BERT, integrasi pemahaman kiri dan kanan terjadi di setiap lapisan neuron, bukan hanya ditempelkan di akhir. Ini memungkinkan pemahaman konteks yang jauh lebih kaya.

### **Kendala pada Model Standar**

Sayangnya, model bahasa kondisional standar (*standard conditional language models*) memiliki batasan teknis: mereka hanya dapat dilatih dalam satu arah saja, yaitu kiri-ke-kanan atau kanan-ke-kiri.

Mengapa kita tidak bisa sekadar membuat model standar menjadi dua arah? Alasannya adalah masalah teknis yang disebut **kebocoran informasi (*information leakage*)**.

* Jika kita menerapkan **pengkondisian dua arah (*bidirectional conditioning*)** pada model standar, hal ini akan memungkinkan setiap kata untuk secara tidak langsung **"melihat dirinya sendiri" (*see itself*)**.
> **Penjelasan "Melihat Diri Sendiri":** Dalam jaringan saraf yang memiliki banyak lapisan (*multi-layered context*), informasi mengalir sangat kompleks. Jika arahnya tidak dibatasi, lapisan yang lebih tinggi bisa "mengintip" informasi target dari lapisan di bawahnya yang sudah memproses kata tersebut.
>> **Akibatnya:** Model akan dapat memprediksi kata target tersebut secara **trivial** (sangat mudah/tanpa usaha kognitif).
> *Analogy:* Bayangkan siswa yang sedang ujian (model) diminta mengisi titik-titik dalam kalimat. Jika ia bisa melihat kunci jawaban (kata target itu sendiri) melalui pantulan cermin atau celah (kebocoran lapisan), ia akan langsung menyalin jawabannya tanpa benar-benar berpikir atau belajar memahami konteks kalimatnya. Proses pelatihan pun menjadi sia-sia karena model "mencontek" alih-alih belajar.
Berikut adalah penjelasan mendalam mengenai mekanisme pelatihan inovatif yang digunakan dalam BERT, diuraikan agar mudah dipahami secara konsep teknis:

### **Mekanisme Pelatihan: Masked Language Model (MLM)**
Untuk melatih representasi dua arah yang mendalam (*deep bidirectional representation*) tanpa mengalami masalah "kebocoran informasi" (di mana model mengintip jawaban), kami menggunakan pendekatan yang sederhana namun efektif. Kami menutup atau menyembunyikan (*mask*) sebagian persentase dari token input secara acak, dan kemudian menugaskan model untuk memprediksi token-token yang disembunyikan tersebut.

#### **Istilah Teknis dan Asal-Usul**
Kami menyebut prosedur ini sebagai **"Masked LM" (MLM)** atau Model Bahasa Bertopeng. Namun, dalam literatur akademis, konsep ini sering disebut sebagai **Tugas Cloze (*Cloze task*)**.
> *Elaborasi:* Istilah ini merujuk pada metode psikolinguistik klasik yang diperkenalkan dalam **[Cloze procedure: A new tool for measuring readability, Taylor et al., 1953]**. Bayangkan ini seperti soal ujian "isian titik-titik" di sekolah, di mana siswa harus menebak kata yang hilang berdasarkan konteks kalimat di sekitarnya.

#### **Proses Teknis: Dari Vektor ke Prediksi**
Secara teknis, proses prediksi ini bekerja sebagai berikut:
1. Model memproses kalimat dan menghasilkan **vektor tersembunyi akhir (*final hidden vectors*)** yang berhubungan dengan token yang ditutupi (mask) tersebut.
> *Penjelasan:* Vektor ini adalah deretan angka matematika yang merepresentasikan pemahaman model tentang "apa yang seharusnya mengisi kekosongan ini" berdasarkan konteks kiri dan kanan.

2. Vektor-vektor ini kemudian dimasukkan ke dalam lapisan **softmax keluaran (*output softmax*)** di atas seluruh kosakata.
> *Penjelasan:* Fungsi "softmax" mengubah angka-angka vektor tadi menjadi probabilitas (persentase kemungkinan). Model akan melihat ke seluruh daftar kata yang diketahuinya (kosakata) dan menghitung kata mana yang memiliki probabilitas tertinggi untuk mengisi bagian yang kosong, persis seperti cara kerja Model Bahasa (LM) standar.

#### **Parameter Eksperimen**
Dalam seluruh eksperimen yang kami lakukan dalam penelitian ini, kami menerapkan aturan yang konsisten: kami menyembunyikan (*mask*) sebanyak **15%** dari seluruh token WordPiece dalam setiap urutan input secara acak.

#### **Perbedaan dengan "Denoising Auto-encoders"**
Pendekatan kami memiliki perbedaan yang signifikan jika dibandingkan dengan metode **[Extracting and composing robust features with denoising autoencoders, Vincent et al., 2008]**.
* **Denoising Auto-encoders:** Metode ini biasanya berusaha untuk merekonstruksi **seluruh** input. Artinya, jika inputnya rusak atau bising, model harus menulis ulang keseluruhan kalimat menjadi versi yang benar.
* **Pendekatan BERT:** Sebaliknya, kami hanya memprediksi kata-kata yang **ditutupi (masked)** saja, bukan merekonstruksi keseluruhan input. Fokus yang lebih sempit ini memaksa model untuk benar-benar memahami konteks spesifik dari kata yang hilang tersebut.

### **Tantangan: Ketidakcocokan Antara Pelatihan dan Penerapan**
Meskipun metode *Masked Language Model* (MLM) memungkinkan kita untuk mendapatkan model pra-pelatihan yang bersifat dua arah (*bidirectional*), metode ini memiliki satu kelemahan atau sisi negatif.

Kelemahannya adalah terciptanya **ketidakcocokan (*mismatch*)** antara fase pra-pelatihan (*pre-training*) dengan fase penyesuaian halus (*fine-tuning*).
> **Penyebab:** Selama pra-pelatihan, model sangat sering melihat token simbol `[MASK]` dan belajar memprediksi kata di baliknya. Namun, saat model ini digunakan untuk tugas nyata (fase *fine-tuning*), simbol `[MASK]` tersebut **tidak pernah muncul** sama sekali dalam teks.
>> **Analogi:** Bayangkan seorang siswa belajar mengemudi mobil yang kaca depannya selalu ditutupi stiker "TEBAK JALAN" sebagian (pra-pelatihan). Namun, saat ujian SIM yang sebenarnya (fine-tuning), stiker itu dicopot. Jika siswa hanya terbiasa mengemudi dengan stiker, ia mungkin bingung atau tidak optimal saat stiker itu hilang.

### **Solusi Mitigasi: Strategi Pengacakan Cerdas**

Untuk mengurangi dampak masalah ini, kami menerapkan strategi di mana kami **tidak selalu** mengganti kata-kata yang ditargetkan dengan simbol `[MASK]` yang eksplisit.

Prosesnya bekerja sebagai berikut:
Generator data pelatihan akan memilih **15%** dari posisi token (kata) secara acak untuk dijadikan target prediksi.

Namun, jika sebuah token ke- terpilih, kami tidak langsung menutupnya begitu saja. Kami menerapkan tiga skenario penggantian dengan probabilitas tertentu:

1. **Skenario Masking (80%):**
* Sebanyak 80% dari waktu, kami mengganti token tersebut dengan simbol **`[MASK]`**.
> *Contoh:* Kalimat "buku saya merah" menjadi "buku saya `[MASK]`".
>> *Tujuan:* Ini adalah metode standar untuk melatih kemampuan prediksi dua arah.

2. **Skenario Kata Acak (10%):**
* Sebanyak 10% dari waktu, kami mengganti token tersebut dengan **token acak lain**.
> *Contoh:* Kalimat "buku saya merah" menjadi "buku saya *apel*".
>> *Tujuan:* Ini mengajarkan model untuk tidak percaya buta pada setiap kata yang ia lihat, melainkan harus selalu memeriksa konteks. Model harus sadar bahwa kata "apel" di situ salah secara konteks dan memprediksi kata aslinya ("merah").


3. **Skenario Tanpa Perubahan (10%):**
* Sebanyak 10% dari waktu, kami membiarkan token ke- tersebut **tidak berubah** (tetap kata aslinya).
> *Contoh:* Kalimat "buku saya merah" tetap menjadi "buku saya merah".
>> *Tujuan:* Ini untuk mendekatkan kondisi latihan dengan kondisi nyata (*fine-tuning*), di mana kata-kata yang dilihat adalah kata asli. Model tetap harus melakukan prediksi terhadap kata tersebut untuk memperkuat pemahamannya.

### **Proses Pelatihan Akhir**

Setelah salah satu dari tiga skenario di atas terjadi, representasi vektor dari token ke- () akan digunakan untuk memprediksi token aslinya.

Kami mengukur akurasi prediksi ini menggunakan metode **kerugian entropi silang (*cross entropy loss*)**.
> *Penjelasan:* Ini adalah rumus matematika untuk menghitung seberapa jauh tebakan model meleset dari jawaban yang benar. Semakin kecil nilai kerugiannya, semakin akurat modelnya.

Kami juga membandingkan berbagai variasi dari prosedur ini (misalnya mengubah persentase probabilitasnya) yang detailnya dapat dilihat pada **Lampiran C.2** dalam dokumen penelitian ini.

### **Tugas #2: Prediksi Kalimat Berikutnya (*Next Sentence Prediction/NSP*)**
Banyak tugas hilir yang penting dalam dunia pemrosesan bahasa, seperti **Sistem Tanya Jawab (*Question Answering/QA*)** dan **Inferensi Bahasa Alami (*Natural Language Inference/NLI*)**, berlandaskan pada pemahaman mengenai hubungan antara dua kalimat.
> **Masalah pada Model Lama:** Pemahaman hubungan antar-kalimat ini tidak dapat ditangkap secara langsung oleh pemodelan bahasa standar (yang hanya memprediksi kata demi kata).
>> **Solusi:** Agar kami dapat melatih model yang benar-benar memahami hubungan antar kalimat, kami melakukan pra-pelatihan untuk sebuah tugas "biner" (dua pilihan) yaitu memprediksi kalimat berikutnya.

### **Mekanisme Pembangkitan Data**
Kelebihan dari tugas ini adalah data pelatihannya dapat dibangkitkan secara "trivial" (sangat mudah dan otomatis) dari korpus satu bahasa (*monolingual corpus*) manapun, tanpa perlu bantuan manusia.

Secara spesifik, saat kami memilih kalimat A dan kalimat B untuk setiap contoh pra-pelatihan, kami menggunakan aturan pembagian 50:50 sebagai berikut:
1. **Skenario "IsNext" (50% Waktu):**
* Sebanyak 50% dari waktu, kalimat B adalah kalimat yang **sebenarnya** muncul tepat setelah kalimat A di dalam teks asli.
> Contoh:
> Kalimat A: "Pria itu pergi ke toko."
> Kalimat B: "Dia membeli sebotol susu."
> Label: *IsNext* (Benar ini lanjutannya).

2. **Skenario "NotNext" (50% Waktu):**
* Sebanyak 50% dari waktu lainnya, kalimat B adalah **kalimat acak** yang diambil dari bagian lain dalam korpus.
> Contoh:
> Kalimat A: "Pria itu pergi ke toko."
> Kalimat B: "Penguin adalah burung yang tidak bisa terbang."
> Label: *NotNext* (Bukan ini lanjutannya).

### **Peran Vektor C dan Hasil Kinerja**

Seperti yang ditunjukkan pada **Gambar 1** dalam dokumen, token `C` (representasi dari token spesial `[CLS]`) digunakan untuk melakukan prediksi kalimat berikutnya (NSP) ini.
> **Akurasi Tinggi:** Model akhir yang kami hasilkan berhasil mencapai akurasi sebesar **97% hingga 98%** pada tugas NSP ini.
>> **Dampak Signifikan:** Meskipun tugas ini terlihat sangat sederhana (hanya tebak "nyambung" atau "tidak"), kami mendemonstrasikan di **Bagian 5.1** bahwa pra-pelatihan dengan tugas ini sangat bermanfaat bagi kinerja tugas Tanya Jawab (QA) dan Inferensi Bahasa (NLI).
>>> **Catatan Teknis Penting:** Vektor `C` itu sendiri bukanlah representasi kalimat yang bermakna jika berdiri sendiri tanpa *fine-tuning*. Hal ini dikarenakan vektor tersebut dilatih khusus hanya untuk memahami probabilitas NSP, bukan makna semantik kalimat secara umum.

### **Hubungan dengan Penelitian Terdahulu**
Tugas **Prediksi Kalimat Berikutnya (NSP)** yang kami gunakan memiliki hubungan yang erat dengan tujuan pembelajaran representasi (*representation-learning objectives*) yang telah digunakan dalam penelitian-penelitian sebelumnya, khususnya dalam **[Discourse-based objectives for fast unsupervised sentence representation learning, Jernite et al., 2017]** dan **[An efficient framework for learning sentence representations, Logeswaran and Lee, 2018]**.

Meskipun memiliki kemiripan konsep dasar, terdapat perbedaan fundamental dalam cara **transfer pengetahuan** dilakukan antara pekerjaan terdahulu tersebut dengan BERT.

#### **Perbedaan Metode Transfer**
1. **Pendekatan Pekerjaan Terdahulu (Transfer Parsial):**
Dalam penelitian-penelitian sebelumnya, hanya **penyematan kalimat (*sentence embeddings*)** saja yang ditransfer ke tugas-tugas hilir (*down-stream tasks*).
> *Penjelasan:* Ini berarti mereka hanya mengambil "ringkasan kode" dari kalimat yang dihasilkan oleh model, lalu kode itu dipakai sebagai fitur tambahan di model baru. Struktur otak (parameter) model aslinya tidak ikut dibawa atau disesuaikan.


2. **Pendekatan BERT (Transfer Total):**
Sebaliknya, BERT melakukan transfer terhadap **semua parameter** untuk menginisialisasi parameter model tugas akhir (*end-task model parameters*).
> *Penjelasan:* BERT tidak hanya mengirimkan ringkasan, tetapi "memindahkan seluruh otak" model yang sudah dilatih. Saat menghadapi tugas baru, seluruh bobot dan koneksi saraf yang sudah dipelajari saat pra-pelatihan digunakan sebagai titik awal (*initialization*), lalu semuanya disesuaikan kembali (*fine-tuned*). Ini membuat proses adaptasi jauh lebih mendalam dan menyeluruh.

### **Data Pra-Pelatihan (*Pre-training Data*)**
Prosedur pra-pelatihan yang kami lakukan sebagian besar mengikuti standar literatur yang sudah ada mengenai pra-pelatihan model bahasa.

Untuk "korpus" pra-pelatihan ini, kami menggunakan kombinasi dari dua sumber data raksasa.

> **Konsep "Korpus":** Dalam linguistik komputasi, korpus adalah kumpulan besar teks tertulis atau lisan yang terstruktur dan digunakan sebagai bahan analisis atau pelatihan mesin.

Sumber data tersebut adalah:
1. **BooksCorpus:** Kumpulan teks buku yang berjumlah **800 Juta kata**. Referensi: [Aligning books and movies: Towards story-like visual explanations by watching movies and reading books, Zhu et al., 2015].
2. **Wikipedia Bahasa Inggris:** Kumpulan artikel ensiklopedia yang jauh lebih masif, berjumlah **2.500 Juta kata**.

**Proses Penyaringan Wikipedia**
Khusus untuk data Wikipedia, kami melakukan penyaringan ketat. Kami hanya mengekstrak bagian **teks bacaan (passages)** saja dan secara sengaja mengabaikan daftar (lists), tabel, dan judul (headers). Hal ini dilakukan agar model fokus memepelajari struktur kalimat yang naratif, bukan data tabular.

### **Pentingnya Data Tingkat Dokumen**
Terdapat satu syarat teknis yang sangat krusial dalam pemilihan data ini: Sangatlah kritis untuk menggunakan **korpus tingkat dokumen (*document-level corpus*)**.

Kami menghindari penggunaan **korpus tingkat kalimat yang diacak (*shuffled sentence-level corpus*)**, seperti dataset *Billion Word Benchmark* yang sering digunakan dalam studi lain (Referensi: [One billion word benchmark for measuring progress in statistical language modeling, Chelba et al., 2013]).

**Mengapa Ini Penting?**
Alasannya adalah untuk mendapatkan **urutan yang panjang dan bersambung (*long contiguous sequences*)**.

* *Analogi:* Membaca "korpus tingkat dokumen" ibarat membaca novel dari Bab 1 sampai tamat, sehingga Anda mengerti alur cerita dan hubungan antar-bab. Sebaliknya, membaca "korpus kalimat acak" ibarat membaca potongan kertas nasib (fortune cookies) yang diambil acak dari toples; Anda mungkin mengerti arti kalimatnya satu per satu, tapi Anda tidak akan pernah memahami konteks yang lebih luas atau hubungan sebab-akibat yang panjang. BERT membutuhkan "bacaan novel" tersebut untuk memahami konteks yang mendalam.

## 3.2 Fine-tuning BERT
Proses penyesuaian halus (*fine-tuning*) pada BERT sebenarnya sangatlah lugas dan langsung (*straightforward*). Kemudahan ini dimungkinkan oleh mekanisme **atensi mandiri (*self-attention*)** yang terdapat dalam arsitektur Transformer.

Mekanisme ini memberikan fleksibilitas luar biasa bagi BERT untuk memodelkan berbagai macam tugas hilir (*downstream tasks*)—baik itu tugas yang melibatkan teks tunggal maupun pasangan teks—hanya dengan cara menukar input dan output yang sesuai.
> **Penjelasan Fleksibilitas:** Bayangkan BERT sebagai mesin pengolah universal. Anda tidak perlu membongkar mesinnya untuk pekerjaan berbeda. Jika Anda ingin menganalisis satu kalimat, masukkan satu kalimat. Jika ingin membandingkan dua kalimat, masukkan keduanya. Mesin (mekanisme atensi) akan menyesuaikan diri secara otomatis.

### **Perbandingan Strategi Pengolahan Pasangan Teks**

Untuk aplikasi yang melibatkan pemrosesan **pasangan teks** (seperti tanya-jawab atau membandingkan kemiripan dua kalimat), terdapat perbedaan mendasar antara cara kerja model umum dengan cara kerja BERT.

#### **1. Pola Umum (Pendekatan Terdahulu)**

Pola yang umum digunakan dalam penelitian sebelumnya adalah mengodekan pasangan teks secara **independen** terlebih dahulu, baru kemudian menerapkan "atensi silang dua arah" (*bidirectional cross attention*).
> **Proses:** Komputer membaca Teks A sendirian dan membuat ringkasannya. Lalu, komputer membaca Teks B sendirian dan membuat ringkasannya. Setelah itu, baru kedua ringkasan tersebut dipertemukan untuk dicari hubungannya.
>> **Referensi:** Pendekatan ini terlihat pada studi **[A decomposable attention model for natural language inference, Parikh et al., 2016]** dan **[Bidirectional attention flow for machine comprehension, Seo et al., 2017]**.

#### **2. Pendekatan BERT (Penyatuan Tahapan)**

Sebaliknya, BERT menggunakan mekanisme atensi mandiri untuk **menyatukan (*unify*)** kedua tahapan tersebut (pengodean dan pencocokan).
> **Proses Konkatenasi:** BERT menggabungkan (*concatenated*) pasangan teks tersebut menjadi satu urutan panjang sejak awal.
>> **Efektivitas:** Karena kedua teks sudah digabung dalam satu wadah, saat BERT melakukan *self-attention*, ia secara otomatis melakukan "atensi silang dua arah" antara kedua kalimat tersebut.
>>> **Analogi:** Jika metode lama ibarat membaca dua buku di dua ruangan berbeda lalu membandingkan catatan, metode BERT ibarat meletakkan kedua buku di meja yang sama dan menarik garis merah antar-halaman yang berhubungan secara langsung dan bersamaan. Ini membuat pemahaman hubungan antar-kalimat menjadi jauh lebih efisien.

Berikut adalah penjelasan ulang mengenai mekanisme teknis penerapan BERT pada berbagai tugas spesifik, yang diuraikan dengan analogi dan penjelasan konsep agar mudah dipahami:

### **Mekanisme "Plug-and-Play" untuk Tugas Spesifik**

Untuk setiap tugas bahasa yang berbeda, kami menerapkan pendekatan yang sangat sederhana: kami cukup "mencolokkan" (*plug in*) input dan output yang spesifik untuk tugas tersebut ke dalam model BERT. Selanjutnya, kami melakukan **fine-tuning** (penyesuaian halus) terhadap **seluruh** parameter model dari ujung ke ujung (*end-to-end*).
> *Penjelasan:* Artinya, kita tidak hanya melatih lapisan terluarnya saja. Saat model belajar tugas baru, seluruh "otak" BERT ikut menyesuaikan diri sedikit demi sedikit untuk mengoptimalkan kinerja pada tugas tersebut.

### **Adaptasi Input: Fleksibilitas "Kalimat A" dan "Kalimat B"**
Pada bagian input, struktur "Kalimat A" dan "Kalimat B" yang dipelajari model saat pra-pelatihan bersifat sangat fleksibel. Kedua slot ini dapat dianalogikan atau dipetakan ke dalam berbagai format data, tergantung pada tugasnya.

Berikut adalah empat skenario pemetaan input tersebut:
1. **Untuk Tugas Parafrase (*Paraphrasing*):**
Slot A dan B diisi dengan **pasangan kalimat**. Model bertugas mengecek apakah kedua kalimat tersebut memiliki makna yang sama.
2. **Untuk Tugas Entailment (Logika Bahasa):**
Slot A dan B diisi dengan pasangan **Hipotesis dan Premis** (*hypothesis-premise pairs*). Model bertugas menentukan apakah hipotesis tersebut terbukti benar berdasarkan premis yang diberikan.
3. **Untuk Tugas Tanya Jawab (*Question Answering*):**
Slot A dan B diisi dengan pasangan **Pertanyaan dan Paragraf Bacaan** (*question-passage pairs*). Model harus mencari jawaban di dalam paragraf tersebut.
4. **Untuk Klasifikasi Teks atau Penandaan Urutan:**
Slot ini diisi dengan apa yang kami sebut secara teknis sebagai **pasangan "teks-∅" yang terdegenerasi (*degenerate text-∅ pair*)**.
> *Penjelasan:* Istilah rumit ini sebenarnya sederhana. Ini berarti kita hanya memiliki **satu teks** (di slot A), sementara slot B dibiarkan **kosong** (simbol ∅ melambangkan himpunan kosong/null). Ini digunakan misalnya saat kita hanya ingin mengecek sentimen satu kalimat ulasan film.

### **Adaptasi Output: Token vs. Klasifikasi**
Pada bagian output (keluaran), mekanisme BERT juga beradaptasi sesuai jenis tugasnya:
* **Tugas Tingkat Token (*Token-level Tasks*):**
Untuk tugas seperti **penandaan urutan** (misalnya: menentukan mana kata benda, mana kata kerja) atau **tanya jawab** (menandai awal dan akhir jawaban), representasi dari **setiap token** (kata) akan dimasukkan ke dalam lapisan output. Model membuat keputusan untuk setiap kata secara individu.

* **Tugas Klasifikasi (*Classification Tasks*):**
Untuk tugas seperti **entailment** atau **analisis sentimen** (misalnya: menentukan apakah email ini "spam" atau "bukan"), kami hanya menggunakan representasi dari token spesial **`[CLS]`**. Representasi ini merangkum pemahaman seluruh kalimat, lalu dimasukkan ke lapisan output untuk menghasilkan satu keputusan label tunggal.

### **Efisiensi Sumber Daya: Pra-Pelatihan vs. Penyesuaian Halus**
Jika dibandingkan dengan proses pra-pelatihan (*pre-training*) yang sangat berat dan memakan waktu, proses penyesuaian halus (*fine-tuning*) relatif jauh lebih **"murah" (*inexpensive*)**.
> *Penjelasan Konsep:* Dalam konteks kecerdasan buatan, kata "murah" di sini bukan hanya soal uang, tetapi merujuk pada **biaya komputasi** (penggunaan listrik, waktu pemrosesan, dan kekuatan perangkat keras). Pra-pelatihan ibarat membesarkan anak dari bayi hingga sarjana (butuh tahunan), sedangkan *fine-tuning* ibarat memberikan kursus singkat selama satu hari agar sarjana tersebut bisa mengerjakan tugas kantor tertentu.

### **Replikasi Hasil Eksperimen**
Efisiensi ini terbukti dari kemudahan dalam mereplikasi atau meniru ulang hasil penelitian kami. Seluruh hasil yang tercantum dalam makalah ini dapat direplikasi dengan waktu yang sangat singkat, asalkan prosesnya dimulai dari **model pra-pelatihan yang sama persis (*exact same pre-trained model*)**.

Berikut adalah estimasi waktu komputasi yang dibutuhkan:
1. **Menggunakan Cloud TPU:** Paling lama hanya memakan waktu **1 jam** pada satu unit Cloud TPU tunggal.
> *Definisi Jargon:* **TPU (*Tensor Processing Unit*)** adalah sirkuit khusus buatan Google yang dirancang spesifik untuk mempercepat tugas-tugas *machine learning*. Ia bekerja jauh lebih cepat daripada chip komputer biasa untuk tugas ini.

2. **Menggunakan GPU:** Hanya memakan waktu **beberapa jam** pada satu unit GPU.
> *Definisi Jargon:* **GPU (*Graphics Processing Unit*)** adalah kartu grafis yang biasa ada di komputer gaming, namun juga sangat andal untuk perhitungan matematika paralel yang dibutuhkan AI.

### **Rincian Implementasi**
Untuk memastikan transparansi dan kemudahan bagi peneliti lain yang ingin mencoba metode kami:
* Kami menguraikan detail spesifik untuk setiap tugas (*task-specific details*) pada sub-bab yang sesuai di dalam **Bagian 4**.
* Informasi yang lebih rinci dan teknis dapat ditemukan pada bagian lampiran, tepatnya di **Appendix A.5**.

## 4 Experiments

### **Hasil Penyesuaian Halus (*Fine-tuning*) BERT**

Pada bagian ini, kami menyajikan hasil performa dari proses *fine-tuning* model BERT terhadap **11 tugas Pemrosesan Bahasa Alami (NLP)** yang berbeda. Evaluasi ini dilakukan untuk membuktikan sejauh mana kemampuan pemahaman bahasa yang didapat selama fase pra-pelatihan dapat diterapkan secara efektif pada berbagai masalah spesifik.

#### **Konteks Pengujian**

Setiap tugas NLP ini dirancang untuk menguji aspek kecerdasan bahasa yang berbeda-beda, mulai dari logika berpikir, pemahaman bacaan, hingga pendeteksian emosi dalam kalimat. Sebagai pengingat, proses *fine-tuning* ini dilakukan dengan mengambil model BERT yang sudah "pintar" secara umum, lalu memberikan latihan tambahan yang sangat singkat (hanya butuh waktu sekitar 1 jam menggunakan infrastruktur tertentu) untuk setiap tugas spesifik tersebut.

#### **Cakupan Evaluasi**

Ke-11 tugas yang kami uji mencakup:

1. **Tugas Tingkat Kalimat:** Menguji kemampuan model dalam memahami makna utuh satu atau sepasang kalimat (seperti menentukan apakah dua kalimat bermakna sama).
2. **Tugas Tingkat Token:** Menguji ketelitian model dalam menganalisis bagian-bagian terkecil dari teks (seperti mencari jawaban di dalam sebuah paragraf atau mengidentifikasi nama orang).

Data-data yang akan dijabarkan pada sub-bab berikutnya akan merujuk pada standar pengukuran global dalam dunia AI, seperti tolak ukur **GLUE** (*General Language Understanding Evaluation*), dataset **SQuAD** untuk tanya jawab, dan data **CoNLL-2003** untuk pengenalan identitas kata.

## 4.1 GLUE (General Language Understanding Evaluation)
Tolak ukur **GLUE (*General Language Understanding Evaluation*)** adalah sebuah koleksi tugas pemahaman bahasa alami yang sangat beragam.

* **Apa itu GLUE?** Bayangkan GLUE sebagai "olimpiade dasalomba" untuk kecerdasan buatan. Daripada hanya menguji lari cepat (satu tugas), GLUE menguji berbagai kemampuan sekaligus—seperti logika, tata bahasa, dan analisis sentimen—untuk melihat siapa atlet (model) yang paling cerdas secara menyeluruh.
> **Referensi:** Standar ini dijelaskan dalam makalah **[GLUE: A multi-task benchmark and analysis platform for natural language understanding, Wang et al., 2018a]**.
>> **Detail Tambahan:** Deskripsi mendetail mengenai kumpulan data (dataset) yang ada di dalam GLUE dapat dilihat pada **Lampiran B.1**.

### **Mekanisme Penyesuaian Halus (*Fine-tuning*) pada GLUE**
Untuk melakukan *fine-tuning* atau melatih model BERT agar jago dalam tugas-tugas GLUE, kami menggunakan pendekatan teknis sebagai berikut:

1. **Representasi Input:**
Kami menyajikan urutan input (baik itu untuk kalimat tunggal maupun pasangan kalimat) persis seperti yang telah dijelaskan pada **Bagian 3** sebelumnya.
> *Kilas Balik:* Input diawali dengan token `[CLS]`, diikuti teks, dan dipisahkan oleh `[SEP]` jika ada dua kalimat.

2. **Agregasi Representasi:**
Kami menggunakan **vektor tersembunyi akhir **.
> *Penjelasan:* Ini adalah kode digital terakhir yang dihasilkan model yang berhubungan dengan token input pertama, yaitu token **`[CLS]`**.
>> *Fungsi:* Token ini berfungsi sebagai **representasi agregat** atau "ringkasan pemahaman" dari keseluruhan input.

3. **Penambahan Parameter Baru:**
Satu-satunya komponen baru yang kami tambahkan selama proses *fine-tuning* ini hanyalah **bobot lapisan klasifikasi ()**.
> *Simbol Matematika:*  melambangkan jumlah label (kategori jawaban), sedangkan  adalah ukuran tersembunyi model.
>> *Analogi:* Jika BERT adalah otak yang sudah pintar,  ini hanyalah "lembar jawaban pilihan ganda" yang kita berikan agar otak tersebut bisa mencentang jawaban yang benar (misalnya: label "Positif" atau "Negatif"). Kita tidak perlu merombak otaknya, cukup beri alat tulis baru.

4. **Perhitungan Kerugian (*Loss Calculation*):**
Kami menghitung standar **kerugian klasifikasi (*classification loss*)** menggunakan data  (ringkasan input) dan  (bobot klasifikasi).
> *Rumus:* Rumus yang digunakan adalah `log(softmax(CWT))`.
>> *Penjelasan Rumus:* Fungsi *softmax* mengubah angka-angka mentah menjadi persentase probabilitas, dan fungsi *log* membantu menghitung seberapa jauh tebakan model meleset dari jawaban benar, agar model bisa mengoreksi dirinya sendiri.

### **Prosedur Teknis Penyesuaian Halus (*Fine-tuning*) untuk GLUE**
Dalam melakukan penyesuaian halus pada seluruh tugas yang tergabung dalam tolak ukur GLUE, kami menetapkan standar prosedur operasional sebagai berikut:

#### **1. Pengaturan Hyperparameter Dasar**
Kami menggunakan **ukuran *batch* (batch size) sebesar 32** dan melatih model selama **3 *epoch*** di atas data.
> **Ukuran Batch (32):** Ini berarti model tidak belajar satu per satu soal, melainkan langsung memproses 32 contoh soal sekaligus dalam satu waktu sebelum memperbarui pemahamannya. Ini mempercepat proses belajar.
>> **3 Epoch:** Satu *epoch* artinya model sudah melihat *seluruh* data latihan satu kali. Jadi, 3 *epoch* berarti model "membaca ulang" seluruh materi pelajaran sebanyak tiga kali putaran penuh untuk memastikan ilmunya meresap.

#### **2. Pemilihan Laju Pembelajaran (*Learning Rate*)**
Untuk setiap tugas yang berbeda, kami melakukan seleksi untuk memilih **laju pembelajaran** yang paling optimal. Kami memilih satu yang terbaik di antara empat kandidat angka: **5e-5, 4e-5, 3e-5, dan 2e-5**. Pemilihan ini didasarkan pada kinerja model di **Set Pengembangan (*Dev set*)**.
> **Laju Pembelajaran:** Bayangkan ini sebagai seberapa besar "langkah" yang diambil model saat memperbaiki kesalahannya. Jika langkahnya terlalu besar (angka tinggi), ia mungkin melompati jawaban yang benar. Jika terlalu kecil (angka rendah), ia belajar terlalu lambat. Kami mencoba variasi angka kecil (notasi ilmiah 5e-5 artinya 0.00005) untuk mencari langkah yang paling pas.
>> **Dev Set:** Ini adalah kumpulan soal latihan "uji coba" yang punya kunci jawaban, digunakan untuk menyetel kemampuan model sebelum ujian akhir yang sebenarnya.

#### **3. Mengatasi Ketidakstabilan pada Model Besar**
Kami menemukan fenomena khusus pada model ****: proses *fine-tuning* terkadang menjadi **tidak stabil (*unstable*)** ketika diterapkan pada dataset yang berukuran kecil.
> *Arti Tidak Stabil:* Hasilnya bisa sangat bervariasi; kadang bagus sekali, kadang buruk sekali, meskipun pengaturannya mirip. Ini terjadi karena model yang "otaknya" sangat besar (Large) mudah bingung atau *overthinking* jika datanya terlalu sedikit.
>> **Solusi: Mulai Ulang Acak (*Random Restarts*)**
Untuk mengatasi hal ini, kami melakukan beberapa kali **mulai ulang acak (*random restarts*)** dan kemudian memilih model dengan hasil terbaik berdasarkan skor di *Dev set*.

Dalam prosedur *random restart* ini, prinsipnya adalah:
1. **Yang Tetap Sama:** Kami tetap menggunakan **pos pemeriksaan pra-pelatihan (*pre-trained checkpoint*)** yang sama. Artinya, "pengetahuan dasar" atau otak awal modelnya tidak diganti.
2. **Yang Diubah:**
* **Pengacakan Data (*Data Shuffling*):** Kami mengubah urutan data latihan. Ini ibarat mengubah urutan soal agar model tidak menghafal pola urutan.
* **Inisialisasi Lapisan Klasifikasi:** Kami mereset angka-angka awal pada lapisan penentu jawaban (*classifier layer*) secara acak. Ini ibarat memberikan lembar jawaban kosong baru untuk memulai lagi dari awal yang segar.

#### **4. Prosedur Evaluasi Akhir**
Perlu dicatat bahwa distribusi dataset GLUE yang kami terima **tidak menyertakan label Tes (*Test labels*)**.
> *Penjelasan:* Kunci jawaban untuk ujian akhir dirahasiakan oleh penyelenggara GLUE agar peserta tidak bisa curang.

Oleh karena itu, kami hanya mengirimkan **satu kali penyerahan (*single submission*)** ke server evaluasi GLUE untuk masing-masing model, baik itu  maupun . Ini menjamin integritas hasil bahwa kami tidak mencoba menebak-nebak kunci jawaban berkali-kali.


### **Analisis Hasil Eksperimen pada Tabel 1**
Seluruh data hasil kinerja model kami dirangkum dan disajikan secara lengkap dalam **Tabel 1**. Berdasarkan data tersebut, terlihat dominasi yang sangat jelas dari kedua varian model kami.

#### **Keunggulan Menyeluruh Di Atas Pesaing**
Baik model **** maupun **** berhasil mengungguli **semua sistem lain** pada **seluruh tugas** yang diujikan.

* **Margin Substansial:** Kemenangan ini bukan kemenangan tipis, melainkan dengan selisih atau margin yang **substansial (sangat signifikan)**.
* **Peningkatan Rata-rata:** Jika dibandingkan dengan status *state-of-the-art* (rekor kinerja terbaik) sebelumnya, model kami mencatatkan peningkatan akurasi rata-rata sebagai berikut:
* **:** Mencapai peningkatan sebesar **4.5%**.
* **:** Mencapai peningkatan sebesar **7.0%**.

#### **Perbandingan Kritis: BERT vs. OpenAI GPT**

Ada satu catatan teknis yang sangat penting untuk diperhatikan mengenai perbandingan ini. Secara arsitektur model, **** dan **OpenAI GPT** sebenarnya **hampir identik**.

* *Mengapa ini penting?* Ini menunjukkan bahwa perbedaan performa bukan disebabkan oleh ukuran atau bentuk dasar model, melainkan oleh satu faktor pembeda utama: **pengatopan atensi (*attention masking*)**.
* *Penjelasan:* OpenAI GPT menggunakan pengatopan yang membatasi pandangan hanya ke kiri (satu arah), sedangkan BERT menggunakan mekanisme yang memungkinkan pandangan dua arah. Perbedaan cara "melihat" inilah yang menjadi kunci lonjakan performa tersebut.

#### **Kinerja pada Tugas MNLI**
Kami menyoroti hasil pada tugas **MNLI (*Multi-Genre Natural Language Inference*)**.
> *Konteks:* MNLI adalah tugas terbesar dan yang paling sering dijadikan rujukan utama dalam tolak ukur GLUE untuk menguji kemampuan logika bahasa.
>> *Hasil:* Pada tugas krusial ini, BERT berhasil memperoleh peningkatan akurasi absolut sebesar **4.6%** dibandingkan rekor sebelumnya.

#### **Skor Papan Peringkat (Leaderboard) Resmi**
Pada papan peringkat atau *leaderboard* resmi GLUE, dominasi BERT terlihat sangat mencolok jika dibandingkan dengan pesaing terdekatnya saat makalah ini ditulis:
* **Skor :** Mencapai skor total **80.5**.
* **Skor OpenAI GPT:** Mencapai skor total **72.8**.

Perbedaan skor yang jauh ini menegaskan bahwa pendekatan dua arah BERT membawa dampak revolusioner dibandingkan pendekatan satu arah yang digunakan GPT saat itu.

### **Dominasi Model Besar ()**
Berdasarkan hasil eksperimen, kami menemukan pola yang konsisten: model ** secara signifikan mengungguli **.

Keunggulan ini tidak hanya terjadi pada satu atau dua kasus, melainkan terjadi secara merata di **seluruh tugas** yang diujikan.

#### **Keunggulan pada Data Minim**
Temuan yang paling menarik perhatian adalah mengenai korelasi ukuran model dengan jumlah data. Keunggulan  menjadi **terasa paling istimewa dan mencolok** justru pada tugas-tugas yang memiliki **data pelatihan yang sangat sedikit (*very little training data*)**.
> *Implikasi:* Biasanya, model yang sangat besar (banyak parameter) dianggap rentan mengalami *overfitting* (terlalu menghafal) jika dilatih dengan data yang sedikit. Namun, temuan ini menunjukkan bahwa berkat pra-pelatihan yang ekstensif, model yang lebih besar justru memiliki kemampuan generalisasi yang lebih baik, bahkan ketika data untuk tugas spesifiknya terbatas.

Analisis yang lebih mendalam dan menyeluruh mengenai efek ukuran model ini akan kami bahas secara khusus pada **Bagian 5.2**.

https://gluebenchmark.com/leaderboard

## 4.2 SQuAD v1.1

**Stanford Question Answering Dataset (SQuAD v1.1)** adalah sebuah koleksi data masif yang terdiri dari **100.000 pasang pertanyaan dan jawaban**.
* **Asal Data:** Pasangan data ini diperoleh melalui metode *crowdsourcing* (bersumber dari kontribusi banyak orang), memastikan variasi bahasa yang alami.
* **Referensi:** Dataset ini diperkenalkan dalam studi **[SQuAD: 100,000+ questions for machine comprehension of text, Rajpurkar et al., 2016]**.

#### **Mekanisme Tugas: Ekstraksi Jawaban**
Tugas yang harus dilakukan oleh model dalam dataset ini adalah simulasi pemahaman bacaan (*reading comprehension*).
1. **Input:** Model diberikan sebuah **pertanyaan** dan sebuah **paragraf bacaan (passage)** dari Wikipedia yang memuat informasi jawabannya.
2. **Tugas:** Model diminta untuk memprediksi **rentang teks jawaban (*answer text span*)** yang ada di dalam paragraf tersebut.
> **Konsep "Rentang Teks" (*Text Span*):**
> Penting untuk dipahami bahwa dalam tugas ini, model **tidak diminta untuk mengarang** atau menulis kalimat jawaban baru sendiri. Sebaliknya, model bertindak seperti stabilo; ia harus mencari dan menandai (mengekstrak) potongan kalimat yang sudah ada di dalam teks bacaan yang menjawab pertanyaan tersebut secara tepat.

Berikut adalah penjelasan teknis mengenai bagaimana BERT diadaptasi untuk tugas Tanya Jawab (SQuAD), diuraikan langkah demi langkah agar mudah dipahami:

### **Implementasi pada Tugas Tanya Jawab (SQuAD)**
Dalam tugas tanya jawab ini, strategi utama kami adalah mengubah masalah pemahaman teks menjadi masalah **pencarian lokasi**. Model tidak diminta menulis jawaban, melainkan mencari koordinat awal dan akhir dari jawaban di dalam teks.

#### **1. Representasi Input: Urutan Terpaket**
Seperti yang diilustrasikan pada **Gambar 1** (dalam dokumen asli), kami menyusun data input dengan cara menggabungkan pertanyaan dan paragraf bacaan menjadi satu kesatuan yang disebut **urutan terpaket tunggal (*single packed sequence*)**.
* **Pertanyaan:** Ditandai menggunakan **Embedding A**.
* **Paragraf Bacaan:** Ditandai menggunakan **Embedding B**.

#### **2. Vektor Penanda Baru (Start & End)**
Selama proses *fine-tuning*, kami hanya memperkenalkan dua vektor baru yang perlu dipelajari oleh model. Ini sangat efisien karena tidak menambah kerumitan arsitektur yang berlebihan. Dua vektor tersebut adalah:
1. **Vektor Awal ():** Bertugas mencari kata pertama dari jawaban.
2. **Vektor Akhir ():** Bertugas mencari kata terakhir dari jawaban.

#### **3. Mekanisme Perhitungan Probabilitas**
Bagaimana cara model menentukan kata mana yang merupakan awal jawaban? Kami menggunakan operasi matematika **produk titik (*dot product*)** diikuti dengan fungsi **softmax**.

Rumus untuk menghitung probabilitas kata ke- sebagai awal jawaban () adalah:

**Penjelasan Rumus:**
* : Adalah representasi vektor dari kata ke- dalam paragraf.
* : Kami mengalikan vektor kata tersebut dengan Vektor Awal (). Semakin tinggi hasil perkaliannya, semakin cocok kata tersebut sebagai awal jawaban.
* **Softmax:** Bagian penyebut () berfungsi untuk menormalisasi angka-angka tersebut menjadi persentase probabilitas (total 100%) di seluruh kata dalam paragraf.

*Rumus analog yang serupa juga digunakan untuk menghitung probabilitas posisi akhir jawaban.*

#### **4. Menentukan Rentang Jawaban Terbaik**

Setelah model menghitung kemungkinan posisi awal dan akhir, model harus memutuskan satu rentang jawaban final.

* **Skor Rentang:** Skor untuk sebuah rentang kandidat (dari posisi  ke posisi ) didefinisikan sebagai penjumlahan skor awal dan skor akhir: .
* **Batasan Logika:** Kami memilih rentang dengan skor maksimum, namun dengan syarat ketat bahwa ****.
* *Penjelasan:* Syarat ini memastikan bahwa kata akhir tidak boleh muncul sebelum kata awal (karena itu tidak logis).

#### **5. Objektif dan Parameter Pelatihan**
* **Tujuan Pelatihan (*Training Objective*):** Model dilatih untuk meminimalkan kesalahan dengan memaksimalkan **jumlah log-likelihood** dari posisi awal dan akhir yang benar.
* **Hyperparameter:** Kami melakukan *fine-tuning* selama **3 epoch** dengan laju pembelajaran (*learning rate*) **5e-5** dan ukuran *batch* **32**.

### **Analisis Hasil dan Strategi Augmentasi Data (Tabel 2)**
Pada **Tabel 2** di dalam dokumen, kami menampilkan perbandingan kinerja antara model BERT dengan berbagai sistem terbaik yang ada saat ini. Perbandingan ini mencakup:
1. **Entri Papan Peringkat (*Leaderboard Entries*):** Skor-skor tertinggi yang tercatat di situs resmi SQuAD.
2. **Sistem Terpublikasi:** Sistem-sistem ternama yang detail teknisnya sudah dipublikasikan dalam makalah ilmiah, seperti sistem karya **Seo et al. (2017)**, **Clark dan Gardner (2018)**, **Peters et al. (2018a)**, dan **Hu et al. (2018)**.

#### **Tantangan Kompetisi: Transparansi dan Aturan Data**
Ada dua hal penting yang perlu dicatat mengenai kompetisi di papan peringkat SQuAD:

1. **Kurangnya Dokumentasi Publik:**
Banyak sistem yang menduduki peringkat teratas di papan peringkat **tidak memiliki deskripsi sistem publik yang mutakhir**.
> *Implikasi:* Mereka seringkali merupakan "kotak hitam" atau sistem rahasia perusahaan/tim peneliti yang belum membuka resep dapurnya, sehingga sulit untuk dibandingkan secara adil dari segi arsitektur.

2. **Izin Penggunaan Data Eksternal:**
Aturan kompetisi memperbolehkan peserta untuk menggunakan **data publik apa saja** saat melatih sistem mereka. Ini berarti peserta boleh menambah "bahan belajar" model dari sumber lain di luar SQuAD untuk mendongkrak skor.

#### **Strategi BERT: Augmentasi Data Moderat**
Merespons aturan tersebut, kami memutuskan untuk menggunakan strategi **augmentasi data yang moderat (*modest data augmentation*)** pada sistem kami agar tetap kompetitif.

Strategi kami adalah melakukan **penyesuaian halus bertingkat**:
1. **Tahap Pemanasan (TriviaQA):** Pertama-tama, kami melatih model pada dataset **TriviaQA** (Joshi et al., 2017).
> *Tujuan:* Ini ibarat memberikan latihan soal umum terlebih dahulu kepada siswa agar terbiasa dengan format tanya-jawab yang lebih luas.

2. **Tahap Spesialisasi (SQuAD):** Setelah itu, barulah kami melakukan *fine-tuning* pada dataset target, yaitu **SQuAD**.

Dengan cara ini, model sudah memiliki "otot" tanya-jawab yang lebih kuat sebelum masuk ke pelatihan spesifik SQuAD.

### **Pencapaian Kinerja Terbaik**
Sistem terbaik yang kami kembangkan berhasil memecahkan rekor dan mengungguli sistem teratas di papan peringkat (*leaderboard*) saat itu dengan margin yang sangat meyakinkan:
* **Mode Ensemble (Gabungan):** Unggul sebesar **+1.5 skor F1**.
* **Mode Sistem Tunggal (*Single System*):** Unggul sebesar **+1.3 skor F1**.
> **Catatan Konsep:** **Skor F1** adalah ukuran rata-rata harmonis antara presisi (ketepatan) dan *recall* (kelengkapan) jawaban. Semakin tinggi skornya, semakin sempurna jawaban yang diberikan model.

#### **Dominasi Model Tunggal vs. Ensemble**
Terdapat satu fakta yang sangat mengejutkan dari hasil ini: **Model BERT tunggal kami mampu mengungguli sistem *ensemble* terbaik dari kompetitor.**
> *Mengapa ini hebat?* Sistem *ensemble* biasanya menggabungkan kekuatan dari banyak model sekaligus untuk mendapatkan skor tinggi (berat di komputasi). Fakta bahwa satu unit BERT sendirian bisa mengalahkan "keroyokan" model lain menunjukkan betapa superiornya arsitektur dasar BERT.

#### **Analisis Dampak Data Tambahan (TriviaQA)**
Meskipun kami menggunakan data tambahan dari TriviaQA (seperti dijelaskan sebelumnya), kami melakukan pengujian untuk melihat seberapa bergantung BERT pada data tersebut.
* **Hasil Tanpa TriviaQA:** Jika kami menghilangkan data TriviaQA dari proses pelatihan, skor F1 hanya turun sangat sedikit, yaitu berkisar antara **0.1 hingga 0.4 poin**.
* **Kesimpulan:** Bahkan tanpa bantuan data eksternal pun, BERT masih mengungguli semua sistem yang ada dengan margin yang lebar. Ini membuktikan bahwa kecerdasan utamanya berasal dari pra-pelatihan BERT itu sendiri, bukan sekadar trik penambahan data.
> **Detail Teknis Data TriviaQA:**
> Data TriviaQA yang kami gunakan terdiri dari paragraf-paragraf yang diambil dari *TriviaQA-Wiki*. Kami memotongnya menjadi **400 token pertama** dalam dokumen yang mengandung setidaknya satu jawaban yang mungkin.

## 4.3 SQuAD v2.0
Tugas **SQuAD 2.0** merupakan pengembangan yang lebih menantang dari versi sebelumnya (SQuAD 1.1). Perbedaannya terletak pada unsur **realisme**:
* **SQuAD 1.1:** Asumsinya adalah jawaban *pasti ada* di dalam teks.
* **SQuAD 2.0:** Memperbolehkan kemungkinan bahwa **tidak ada jawaban singkat** di dalam paragraf yang disediakan.
Ini membuat masalah menjadi lebih sulit karena model tidak hanya harus mencari jawaban, tetapi juga harus cukup cerdas untuk berkata, "Saya tidak tahu" atau "Jawabannya tidak ada di sini."

### **Pendekatan Teknis: Strategi "Pos Kosong"**

Untuk menangani tantangan ini, kami memperluas model BERT SQuAD v1.1 dengan pendekatan yang sederhana namun cerdik. Kami memanfaatkan token spesial **`[CLS]`** sebagai "tempat parkir" untuk pertanyaan yang tidak memiliki jawaban.

1. **Mekanisme "Tanpa Jawaban":**
Kami memperlakukan pertanyaan yang tidak memiliki jawaban seolah-olah memiliki "rentang jawaban" yang dimulai dan diakhiri tepat pada token **`[CLS]`**.
> *Analogi:* Bayangkan token `[CLS]` sebagai kotak "N/A" (Not Applicable) pada formulir. Jika model tidak menemukan jawaban di teks, ia akan menunjuk ke kotak ini.

2. **Perluasan Ruang Probabilitas:**
Secara matematis, ruang pencarian untuk posisi awal dan akhir jawaban diperluas untuk mencakup posisi token `[CLS]` tersebut.

### **Logika Prediksi dan Skor Ambang Batas**
Bagaimana model memutuskan antara memberikan jawaban teks atau memilih "tidak ada jawaban"? Kami menggunakan mekanisme perbandingan skor.

Kami membandingkan dua nilai skor:

1. **Skor Tanpa-Jawaban ():**
> (Skor jika awal dan akhir jawaban menunjuk ke token `[CLS]`)*.
2. **Skor Jawaban Terbaik ():**
> (Skor tertinggi dari rentang teks nyata yang ditemukan dalam paragraf)*.

**Keputusan Akhir:**
Kami memprediksi bahwa jawaban **ada (non-null)** hanya jika skor jawaban terbaik tersebut mengalahkan skor tanpa-jawaban ditambah sebuah **ambang batas ()**:
> **Konsep Ambang Batas ():** Nilai  dipilih berdasarkan kinerja di set pengembangan (*Dev set*) untuk memaksimalkan skor F1.
>> *Analogi:*  adalah "standar keyakinan". Model hanya boleh menjawab jika ia *sangat yakin* (skornya jauh lebih tinggi daripada skor kosong). Jika ragu-ragu (selisihnya kecil), model lebih baik memilih diam (tidak menjawab).

### **Detail Pelatihan dan Hasil**
Berbeda dengan SQuAD 1.1, untuk model ini kami **tidak menggunakan** data TriviaQA.
* **Parameter:** *Fine-tuning* dilakukan selama **2 epoch** dengan laju pembelajaran **5e-5** dan ukuran *batch* **48**.

**Hasil Kinerja (Tabel 3):**
Hasil perbandingan dengan entri papan peringkat terdahulu dan karya terpublikasi terbaik (seperti Sun et al., 2018; Wang et al., 2018b) menunjukkan keunggulan BERT.
* Kami mengecualikan sistem lain yang sudah menggunakan BERT sebagai komponennya (untuk perbandingan yang adil dengan teknologi pra-BERT).
* Hasilnya, kami mencatat peningkatan skor F1 sebesar **+5.1** di atas sistem terbaik sebelumnya. Ini adalah lompatan kinerja yang sangat signifikan untuk tugas sesulit SQuAD 2.0.

## 4.4 SWAG (Situations With Adversarial Generations)
Dataset **SWAG (*Situations With Adversarial Generations*)** adalah kumpulan data masif yang berisi **113.000 contoh penyelesaian pasangan kalimat**.
> **Tujuan:** Dataset ini dirancang untuk mengevaluasi **inferensi akal sehat yang membumi (*grounded commonsense inference*)**.
>> **Mekanisme Tugas:** Diberikan sebuah kalimat awal, tugas model adalah memilih **kelanjutan yang paling masuk akal** di antara empat pilihan jawaban yang tersedia.
>>> *Analogi:* Ini seperti tes logika kehidupan sehari-hari. Contoh: "Seseorang sedang bermain gitar di panggung. Apa yang terjadi selanjutnya?" (A) Dia melompat ke penonton, (B) Dia memakan gitarnya. Model harus tahu mana yang secara fisik dan logika sosial mungkin terjadi.

#### **Teknik Penyesuaian Halus (*Fine-tuning*) untuk Pilihan Ganda**
Tantangan teknis di sini adalah bagaimana cara BERT memproses empat pilihan ganda sekaligus. Kami menyelesaikannya dengan mengonstruksi **empat urutan input terpisah**.

Setiap urutan input adalah hasil penggabungan (*concatenation*) antara:
1. **Kalimat Awal (Kalimat A)**
2. **Salah satu Pilihan Kelanjutan (Kalimat B)**

**Proses Penilaian:**
1. Keempat urutan tersebut dimasukkan ke BERT secara paralel.
2. Kami mengambil representasi token **`[CLS]`** () dari setiap urutan.
3. Satu-satunya parameter baru yang kami tambahkan adalah sebuah vektor tugas khusus.
4. Kami menghitung **produk titik (*dot product*)** antara vektor tugas ini dengan representasi  untuk mendapatkan skor bagi setiap pilihan.
5. Skor-skor tersebut dinormalisasi menggunakan lapisan **softmax** untuk menentukan pilihan mana yang memiliki probabilitas tertinggi.

#### **Detail Pelatihan dan Lonjakan Hasil**
Kami melatih model selama **3 epoch** dengan laju pembelajaran **2e-5** dan ukuran *batch* **16**.

Hasil yang disajikan dalam **Tabel 4** menunjukkan lompatan kinerja yang luar biasa:
* **Melampaui ESIM+ELMo:**  mengungguli sistem dasar penulis dataset (ESIM+ELMo) dengan selisih **+27.1%**. Ini adalah peningkatan yang sangat masif dalam standar penelitian AI.
* **Melampaui OpenAI GPT:**  juga mengungguli model canggih OpenAI GPT sebesar **8.3%**.

## 5 Abblation Studies
Pada bagian ini, kami melakukan serangkaian eksperimen yang disebut **studi ablasi** terhadap berbagai aspek dan komponen penyusun BERT.

#### **Konsep Dasar: Apa itu Studi Ablasi?**
Dalam istilah penelitian kecerdasan buatan (AI) dan pembelajaran mesin, "ablasi" berarti metode pengujian dengan cara **menghapus atau mematikan** satu per satu komponen dari sebuah sistem untuk melihat dampaknya.
> **Analogi:** Bayangkan Anda membuat kue yang sangat enak dengan 5 bahan rahasia. Untuk mengetahui bahan mana yang paling membuat kue itu enak, Anda mencoba memanggang kue itu lagi berkali-kali, tapi setiap kali Anda sengaja **menghilangkan satu bahan**. Jika Anda menghilangkan gula dan kuenya jadi tidak enak, berarti gula itu "sangat penting". Jika Anda menghilangkan vanili dan rasanya hampir sama, berarti vanili "kurang penting".
>> **Penerapan di BERT:** Kami ingin mengetahui apakah kehebatan BERT berasal dari sifat dua arahnya, dari tugas prediksi kalimatnya, atau dari ukurannya. Maka, kami akan "mematikan" fitur-fitur tersebut satu per satu untuk mengujinya.

#### **Tujuan Eksperimen**
Tujuan utama dari eksperimen ini adalah untuk memahami **kepentingan relatif (*relative importance*)** dari setiap komponen. Kami ingin membuktikan secara ilmiah bagian mana dari "resep" BERT yang paling berkontribusi terhadap kinerjanya yang tinggi.

Studi ablasi tambahan yang lebih mendetail juga dapat ditemukan pada **Lampiran C (Appendix C)**.

## 5.1 Effect of Pre-trainikng Tasks
Tujuan utama dari eksperimen ini adalah untuk membuktikan secara ilmiah pentingnya **sifat dua arah yang mendalam (*deep bidirectionality*)** pada BERT.

Untuk membuktikannya, kami membandingkan model  yang asli dengan dua varian model lain. Agar perbandingannya adil (apel-ke-apel), kami menggunakan data pra-pelatihan, skema *fine-tuning*, dan *hyperparameter* yang **sama persis**.

Berikut adalah dua varian model yang kami uji:

#### **1. Model "No NSP" (Tanpa Prediksi Kalimat Berikutnya)**
* **Mekanisme:** Model ini tetap bersifat **dua arah (*bidirectional*)**. Ia dilatih menggunakan metode *Masked LM* (MLM) seperti BERT biasa.
* **Perbedaan:** Model ini **tidak** dilatih untuk tugas "Prediksi Kalimat Berikutnya" (NSP).
* *Tujuan:* Untuk melihat apakah tugas NSP benar-benar membantu pemahaman model, atau apakah metode *Masked LM* saja sudah cukup.

#### **2. Model "LTR & No NSP" (Kiri-ke-Kanan & Tanpa NSP)**
* **Mekanisme:** Ini adalah model yang hanya melihat **konteks kiri saja (*left-context-only*)**. Model ini dilatih menggunakan metode standar *Left-to-Right* (LTR), bukan *Masked LM*.
* **Perbedaan:** Selain arah bacanya cuma satu arah, model ini juga dilatih **tanpa** tugas NSP.
* **Kendala Teknis:** Kami tetap menerapkan batasan "kiri-saja" ini pada fase *fine-tuning*.
* *Alasan:* Jika kami melatihnya secara satu arah (pra-pelatihan) tetapi membebaskannya menjadi dua arah saat *fine-tuning*, akan terjadi ketidakcocokan (*mismatch*) yang justru merusak performa.


* **Perbandingan Kompetitif:** Model varian ini secara langsung dapat disandingkan dengan **OpenAI GPT**. Namun, versi ini lebih adil karena menggunakan dataset pelatihan kami yang lebih besar dan skema teknis kami sendiri.

Berikut adalah analisis mendalam mengenai dampak penghilangan komponen-komponen kunci BERT terhadap kinerjanya, diuraikan poin demi poin untuk memperjelas temuan eksperimen:
### **1. Dampak Penghilangan Tugas NSP (*Next Sentence Prediction*)**
Pertama-tama, kami menguji apa yang terjadi jika tugas "Prediksi Kalimat Berikutnya" (NSP) dihapus dari fase pra-pelatihan (model *No NSP*).
* **Hasil:** Seperti terlihat pada **Tabel 5** (dalam dokumen), penghilangan NSP **merusak kinerja secara signifikan**.
* **Area Terdampak:** Penurunan performa paling terasa pada tugas **QNLI, MNLI, dan SQuAD 1.1**.
* **Kesimpulan:** Ini membuktikan bahwa kemampuan memahami hubungan antar-kalimat (logika implikasi) adalah pondasi penting yang dibangun oleh tugas NSP.

### **2. Dampak Dua Arah vs. Satu Arah (MLM vs. LTR)**
Selanjutnya, kami membandingkan model "No NSP" (yang masih dua arah/MLM) dengan model "LTR & No NSP" (satu arah/kiri-ke-kanan).
* **Dominasi MLM:** Model LTR (satu arah) berkinerja **lebih buruk** dibandingkan model MLM (dua arah) pada **semua tugas**.
* **Penurunan Drastis:** Penurunan paling parah terjadi pada tugas **MRPC** dan **SQuAD**.

#### **Mengapa LTR Gagal Total pada SQuAD?**
Secara intuitif, kegagalan model LTR pada tugas tanya jawab (SQuAD) sangat masuk akal.
* **Masalah Konteks:** Dalam model LTR, saat model memproses sebuah token (kata), ia **tidak memiliki konteks sisi kanan**.
* **Analogi:** Bayangkan Anda harus menebak makna kata "bisa" dalam kalimat, tapi Anda dilarang membaca kata-kata setelahnya. Anda tidak akan tahu apakah itu berarti "racun ular" atau "dapat melakukan". Tanpa melihat kelanjutan kalimat, prediksi token menjadi sangat lemah.

### **3. Upaya Penyelamatan Model LTR (Eksperimen BiLSTM)**
Demi memastikan kami memberikan penilaian yang adil (*good faith attempt*), kami mencoba memperkuat model LTR agar tidak kalah telak. Kami menambahkan lapisan **BiLSTM yang diinisialisasi secara acak** di bagian atas model LTR tersebut.
> **Hasil:** Penambahan BiLSTM memang meningkatkan hasil pada SQuAD secara signifikan.
>> **Tapi...:** Hasil akhirnya **masih jauh lebih buruk** dibandingkan model yang sejak awal sudah dua arah (bidirectional) dari fase pra-pelatihan.
>>> **Efek Samping:** Parahnya lagi, penambahan BiLSTM ini justru merusak kinerja pada tugas-tugas GLUE lainnya.

### **4. Mengapa Tidak Menggabungkan LTR + RTL (Seperti ELMo)?**
Kami menyadari ada pendekatan lain, yaitu melatih dua model terpisah (satu LTR, satu RTL/Kanan-ke-Kiri) lalu menggabungkan keduanya, seperti yang dilakukan oleh ELMo. Namun, kami menolak pendekatan ini karena tiga alasan fatal:
1. **Mahal (Biaya Komputasi):** Melatih dua model terpisah memakan biaya **dua kali lipat** dibandingkan melatih satu model dua arah.
2. **Non-Intuitif untuk Tanya Jawab (QA):** Pada model RTL (Kanan-ke-Kiri), model membaca dari belakang ke depan. Artinya, model membaca jawaban (yang biasanya di belakang) *sebelum* membaca pertanyaan (yang di depan). Model RTL tidak bisa mengondisikan jawaban berdasarkan pertanyaan karena ia belum "melihat" pertanyaannya.
3. **Kurang "Sakti" (Less Powerful):** Penggabungan LTR+RTL secara teknis **kalah kuat** dibandingkan model *deep bidirectional*.
* *ELMo:* Hanya menggabungkan pemahaman kiri dan kanan di lapisan paling akhir.
* *BERT:* Menggunakan konteks kiri dan kanan secara bersamaan **di setiap lapisan (*at every layer*)**.
  
## 5.2 Effect of Model Size
Pada bagian ini, kami mengeksplorasi pertanyaan mendasar: **Apakah memperbesar ukuran model akan selalu meningkatkan akurasi?**

Untuk menjawabnya, kami melatih serangkaian model BERT dengan variasi jumlah lapisan, unit tersembunyi, dan kepala atensi yang berbeda-beda, sementara *hyperparameter* dan prosedur pelatihan lainnya dibiarkan sama.

#### **Temuan Kunci dari Tabel 6**
Hasil eksperimen (berdasarkan rata-rata akurasi *Dev Set* dari 5 kali *restart* acak) menunjukkan dua fenomena menarik:

1. **Hukum "Semakin Besar Semakin Baik":**
Terlihat adanya **peningkatan akurasi yang ketat (*strict accuracy improvement*)** seiring dengan bertambahnya ukuran model. Tren ini berlaku di seluruh empat dataset yang diujikan.
2. **Anomali pada Data Kecil (Kasus MRPC):**
Temuan yang paling mengejutkan terjadi pada dataset **MRPC**.
> *Karakteristik MRPC:* Dataset ini sangat kecil, hanya memiliki **3.600 contoh latih** berlabel.
>> *Teori Umum:* Biasanya, model raksasa yang dilatih pada data super-sedikit akan mengalami *overfitting* (gagal menyeralisir).
>>> *Fakta BERT:* Justru model BERT yang lebih besar memberikan hasil yang lebih baik pada data sekecil ini.

#### **Perbandingan Skala dengan Literatur Terdahulu**
Peningkatan ini terbilang mengejutkan karena kami berhasil mendapatkan kenaikan performa yang signifikan di atas model-model yang ukurannya sudah dianggap "raksasa" dalam standar literatur saat itu.

Sebagai perbandingan skala parameter:
* **Transformer Terbesar (Vaswani et al., 2017):**
Hanya memiliki **100 Juta** parameter ().
* **Transformer Terbesar dalam Literatur Lain (Al-Rfou et al., 2018):**
Memiliki **235 Juta** parameter ().
* **Model Kami ():**
Memiliki **110 Juta** parameter.
* **Model Kami ():**
Memiliki **340 Juta** parameter.

Dominasi  (340M) membuktikan bahwa kita belum menyentuh batas atas kemampuan model bahasa; memperbesar model secara ekstrem masih memberikan "imbalan" berupa akurasi yang lebih tinggi.

### **Paradoks Skala: Besar Itu (Selalu) Lebih Baik?**
Sudah lama diketahui dalam dunia AI bahwa memperbesar ukuran model akan terus meningkatkan kinerja pada **tugas berskala besar**, seperti penerjemahan mesin atau pemodelan bahasa. Hal ini terbukti dari data *LM perplexity* pada tabel kami yang menunjukkan perbaikan konsisten.

Namun, penelitian kami adalah karya pertama yang membuktikan secara meyakinkan bahwa **penskalaan ke ukuran ekstrem** juga menghasilkan peningkatan besar pada **tugas berskala sangat kecil**.

Syarat utamanya adalah: Model tersebut harus sudah **melalui pra-pelatihan yang memadai (*sufficiently pre-trained*)**.

### **Mengapa Peneliti Sebelumnya Ragu?**
Sebelum BERT, peneliti lain meragukan bahwa memperbesar model akan selalu membantu, terutama karena mereka menemukan titik jenuh (*diminishing returns*):

1. **Peters et al. (2018b):** Melaporkan hasil yang beragam (tidak selalu bagus) saat meningkatkan ukuran lapisan *bi-LM* dari dua menjadi empat.
2. **Melamud et al. (2016):** Mencatat bahwa meningkatkan dimensi tersembunyi dari 200 ke 600 memang membantu, tetapi menaikkannya lagi ke **1.000 tidak memberikan peningkatan lebih lanjut**.

### **Hipotesis Kami: Fitur vs. Fine-Tuning**
Mengapa hasil kami berbeda dengan mereka? Kuncinya ada pada **pendekatan** yang digunakan.

* **Penelitian Terdahulu (Pendekatan Berbasis Fitur):**
Mereka menggunakan pendekatan *feature-based* (seperti ELMo), di mana representasi model "dibekukan" dan hanya dijadikan input tambahan. Dalam metode ini, menambah ukuran model sepertinya mencapai batas efektivitas.
* **Pendekatan BERT (Fine-tuning Langsung):**
Kami berhipotesis bahwa ketika model **disesuaikan secara langsung (*fine-tuned directly*)** pada tugas hilir, ceritanya berubah.
Karena BERT hanya menggunakan sedikit sekali parameter tambahan yang diacak (hanya lapisan akhir), model tugas spesifik tersebut dapat **"meminjam" kekuatan penuh** dari representasi pra-pelatihan yang besar dan ekspresif tersebut. Akibatnya, bahkan data tugas hilir yang sangat sedikit pun bisa memanfaatkan kecerdasan model raksasa tanpa mengalami masalah.

## 5.3 Feature-based Approach with BERT
Seluruh hasil eksperimen BERT yang telah kami sajikan sebelumnya menggunakan metode **penyesuaian halus (*fine-tuning*)**.
> *Konsep Fine-tuning:* Kami menambahkan satu lapisan klasifikasi sederhana di ujung model, lalu melatih ulang **seluruh** parameter (baik BERT maupun lapisan baru) secara bersamaan. Ibarat membeli baju jadi, lalu menjahit ulang seluruh bagian baju tersebut agar pas di badan.

Namun, ada pendekatan lain yang disebut **pendekatan berbasis fitur**, di mana kita **mengekstrak fitur tetap (*fixed features*)** dari model pra-pelatihan tanpa mengubah model aslinya. Ibarat membeli baju jadi, dan hanya menambahkan aksesoris tanpa merombak jahitan bajunya.

Pendekatan ini memiliki dua keunggulan strategis:

#### **1. Fleksibilitas Arsitektur**
Tidak semua tugas bahasa dapat dengan mudah dipetakan ke dalam arsitektur *Transformer encoder* (arsitektur dasar BERT).
> *Masalah:* Beberapa tugas mungkin memerlukan struktur model yang sangat spesifik atau kompleks yang tidak cocok dengan format input/output standar BERT.
>> *Solusi:* Dengan pendekatan berbasis fitur, kita bisa mengambil "sari pati" (fitur) kecerdasan dari BERT, lalu memasukkannya ke dalam model arsitektur lain yang lebih cocok untuk tugas tersebut.

#### **2. Efisiensi Komputasi Besar**
Terdapat keuntungan biaya komputasi yang sangat besar dengan pendekatan ini.
> *Masalah:* Menjalankan model BERT yang raksasa setiap kali kita ingin melatih ulang adalah proses yang "mahal" (lambat dan boros memori).
>> *Solusi:* Kita bisa melakukan **pra-komputasi (*pre-compute*)**.
>> 1. Jalankan BERT sekali saja pada data pelatihan.
>> 2. Simpan hasil representasinya (fiturnya) ke dalam disk.
>> 3. Gunakan data fitur yang tersimpan itu berkali-kali untuk melatih model-model kecil (murah) di atasnya.
>>> *Analogi:* Ini ibarat merekam kuliah dosen profesor (BERT) sekali saja. Daripada meminta profesor mengajar ulang setiap kali Anda mau belajar (mahal), Anda cukup memutar ulang rekamannya (murah) untuk membuat catatan-catatan kecil.

### **Studi Kasus: NER CoNLL-2003**
Untuk membandingkan mana yang lebih efektif antara pendekatan *fine-tuning* (ubah semua parameter) dengan pendekatan *berbasis fitur* (ekstrak fitur saja), kami menerapkan BERT pada tugas **Named Entity Recognition (NER)** menggunakan dataset standar CoNLL-2003.
> **Apa itu NER?** Tugas ini meminta komputer untuk membaca teks dan menandai kata mana yang merupakan Nama Orang, Nama Lokasi, Nama Organisasi, dll.

### **Spesifikasi Teknis Input dan Model**
Dalam penerapannya, kami menggunakan pengaturan teknis yang spesifik:
1. **Model WordPiece "Case-Preserving":**
Kami menggunakan model pemecah kata (*WordPiece*) yang **mempertahankan kapitalisasi (*case-preserving*)**.
> *Alasan:* Dalam mendeteksi nama (Entitas), huruf besar sangat penting. "apple" (buah) berbeda dengan "Apple" (perusahaan). Model harus bisa membedakan keduanya.

2. **Konteks Dokumen Maksimal:**
Kami memasukkan konteks dokumen sebanyak mungkin yang disediakan oleh data, agar model memiliki gambaran cerita yang utuh.
3. **Tanpa Lapisan CRF (Sederhana):**
Mengikuti praktik standar, kami memformulasikan ini sebagai tugas penandaan (*tagging task*). Namun, kami **tidak menggunakan lapisan CRF (*Conditional Random Field*)** di bagian output.
> *Konteks:* Biasanya, model NER canggih menggunakan CRF untuk memastikan urutan label logis (misalnya: label "Awal Nama" harus diikuti "Tengah Nama"). BERT tidak memerlukannya, cukup klasifikasi biasa.

### **Aturan Sub-Token Pertama**
Tantangan unik BERT adalah ia memecah satu kata menjadi beberapa potongan (*sub-tokens*). Misalnya, kata "Washington" mungkin utuh, tapi kata yang jarang muncul mungkin dipecah.

Bagaimana cara memberi label pada kata yang terpecah?
Kami menggunakan representasi dari **sub-token pertama** sebagai input ke pengklasifikasi.
> *Mekanisme:* Jika kata "bank" dipecah menjadi "ba" dan "##nk", kami hanya melihat representasi "ba" untuk menentukan apakah itu nama organisasi atau bukan. Prediksi pada sub-token pertama dianggap mewakili label seluruh kata tersebut.

### **Eksperimen Ablasi: Pendekatan Berbasis Fitur**
Untuk menguji apakah kita benar-benar *harus* melakukan *fine-tuning* pada seluruh model, kami melakukan eksperimen balikan menggunakan **pendekatan berbasis fitur**.

Dalam metode ini, kami memperlakukan BERT sebagai "generator kode statis".
1. **Ekstraksi Aktivasi:** Kami mengambil **aktivasi** (sinyal keluaran) dari satu atau beberapa lapisan BERT.
2. **Parameter Beku:** Kami **tidak melakukan fine-tuning** pada parameter BERT sama sekali. Otak BERT dibekukan.
3. **Arsitektur Tambahan:** Sinyal yang diekstrak tersebut (embedding kontekstual) dijadikan input untuk model baru di atasnya, yaitu **BiLSTM 2-lapis dengan dimensi 768** yang diinisialisasi secara acak, sebelum akhirnya masuk ke lapisan klasifikasi.

### **Hasil Eksperimen (Tabel 7)**
Hasil yang kami peroleh sangat mengejutkan dan kompetitif.

* **Kompetitif dengan State-of-the-Art:** Model  dengan metode ini mampu bersaing ketat dengan metode-metode terbaik yang ada saat ini.
* **Strategi Terbaik:** Kinerja tertinggi diperoleh dengan cara **menggabungkan (*concatenating*) representasi token dari 4 lapisan tersembunyi paling atas** dari Transformer.

### **Kesimpulan Efektivitas**
Metode penggabungan 4 lapisan atas ini menghasilkan skor yang luar biasa: hanya tertinggal **0.3 skor F1** di belakang metode *fine-tuning* penuh.
> **Implikasi:** Ini membuktikan fleksibilitas BERT. Model ini sangat efektif baik digunakan sebagai model yang disesuaikan penuh (*fine-tuning*) maupun sebagai pengekstraksi fitur canggih (*feature-based*) untuk model lain, memberikan kebebasan bagi pengembang untuk memilih metode yang paling efisien bagi sumber daya mereka.

## Conclusion

Paragraf ini menyoroti pergeseran paradigma dalam dunia pemrosesan bahasa alami (NLP) dan di mana posisi BERT dalam evolusi tersebut.

#### **1. Pentingnya Pra-Pelatihan (*The Era of Pre-training*)**
Bukti-bukti empiris terbaru menunjukkan bahwa **transfer learning** (pembelajaran transfer) dengan model bahasa telah mengubah cara kerja sistem AI.
* **Temuan:** Pra-pelatihan tanpa pengawasan (*unsupervised pre-training*) yang kaya akan data kini menjadi bagian **integral** (tak terpisahkan) dari banyak sistem pemahaman bahasa.
* **Dampak:** Hal ini memungkinkan tugas-tugas yang memiliki **sumber daya rendah (sedikit data)** untuk tetap mendapatkan kinerja tinggi. Sebelumnya, tugas dengan sedikit data sulit mendapatkan hasil bagus, namun sekarang mereka bisa "membonceng" kecerdasan dari arsitektur mendalam yang sudah dilatih sebelumnya.

#### **2. Kontribusi Utama BERT: Mengapa Ini Berbeda?**
Sebelum BERT, kesuksesan transfer learning tersebut sebagian besar didapat menggunakan arsitektur **satu arah yang mendalam (*deep unidirectional architectures*)**, seperti model yang hanya membaca dari kiri-ke-kanan.

Kontribusi terbesar kami dalam penelitian ini adalah **menggeneralisasi** temuan tersebut ke tingkat selanjutnya, yaitu: **Arsitektur Dua Arah yang Mendalam (*Deep Bidirectional Architectures*)**.

Dengan beralih ke arsitektur dua arah ini, kami memungkinkan **satu model pra-pelatihan yang sama** untuk menaklukkan serangkaian tugas NLP yang sangat luas dengan sukses, tanpa perlu merancang model yang berbeda-beda untuk setiap tugas.

## A Additional Details for BERT

## A.1 Illustration of the Pre-training Tasks
Berikut adalah ilustrasi konkret mengenai prosedur *Masked Language Model* (MLM) yang kami terapkan, menggunakan contoh kalimat sederhana agar mudah dipahami.

### **Prosedur Masking: Contoh Kasus**
Mari kita asumsikan kita memiliki sebuah kalimat input yang tidak berlabel:
> **"my dog is hairy"** (anjing saya berbulu)

Selama proses pengacakan, algoritma memilih **token ke-4** (yaitu kata **"hairy"**) sebagai target untuk diprediksi.

Untuk melatih model agar tangguh, kami tidak selalu menghapus kata tersebut. Berikut adalah simulasi 3 skenario yang mungkin terjadi pada kata "hairy" tersebut berdasarkan aturan probabilitas kami:

* **80% Waktu: Ganti dengan Token `[MASK]**`
Ini adalah skenario paling umum. Kata asli ditutup total.
> Input menjadi: **"my dog is [MASK]"**


* *Tugas Model:* Menebak kata yang hilang hanya berdasarkan konteks "my dog is...".


* **10% Waktu: Ganti dengan Kata Acak**
Kata asli diganti dengan kata lain yang dipilih secara acak dari kamus.
> Input menjadi: **"my dog is apple"**


* *Tugas Model:* Mendeteksi bahwa "apple" tidak masuk akal dalam konteks ini dan memprediksi kata aslinya ("hairy").


* **10% Waktu: Biarkan Tidak Berubah (Asli)**
Kata asli dibiarkan tetap ada apa adanya.
> Input menjadi: **"my dog is hairy"**


* *Tujuan:* Ini bertujuan untuk membiaskan representasi ke arah kata yang sebenarnya diamati. Artinya, model diajarkan untuk tidak selalu curiga bahwa input itu salah atau hilang; terkadang input yang dilihatnya memang sudah benar. Model tetap harus memprosesnya untuk memastikan validitasnya.

Berikut adalah penjelasan mengenai keuntungan logis dari prosedur *masking* yang unik tersebut, diuraikan agar alasan di balik desainnya mudah dipahami:

### **Keuntungan Strategi: Memaksa Kewaspadaan Penuh**
Keuntungan utama dari prosedur acak (80-10-10) ini adalah menciptakan ketidakpastian yang konstruktif bagi model **Transformer encoder**.

* **Masalah Jika Hanya Menggunakan `[MASK]`:** Jika kita selalu mengganti kata target dengan `[MASK]`, model akan menjadi malas. Ia hanya akan fokus pada simbol `[MASK]` dan mengabaikan kata-kata lain yang terlihat "normal".
* **Efek Ketidakpastian:** Dengan strategi campuran ini, model **tidak pernah tahu** kata mana yang sebenarnya sedang diuji:
1. Apakah ini kata asli?
2. Apakah ini kata palsu (random)?
3. Atau apakah ini kata yang harus ditebak?

Akibatnya, model "dipaksa" untuk memelihara **representasi kontekstual terdistribusi (*distributional contextual representation*)** untuk **setiap token input**, bukan hanya untuk token yang tertutup. Model harus selalu waspada dan memeriksa validitas setiap kata berdasarkan konteks sekitarnya.

### **Mengapa Penggantian Acak Tidak Merusak Model?**
Anda mungkin bertanya: *"Apakah mengganti kata dengan kata acak (misalnya 'anjing' jadi 'apel') tidak akan membingungkan model dan membuatnya jadi bodoh?"*

Jawabannya adalah **tidak**, karena frekuensinya sangat kecil. Mari kita hitung matematikanya:

* Kami hanya memilih 15% dari total token untuk diprediksi.
* Dari 15% itu, hanya 10% yang diganti dengan kata acak.
* **Total:** .

Artinya, penggantian acak ini hanya terjadi pada **1.5%** dari seluruh kata dalam teks. Jumlah "gangguan" (*noise*) ini cukup kecil sehingga tidak merusak kemampuan pemahaman bahasa model secara keseluruhan, namun cukup signifikan untuk memaksa model belajar lebih keras.

Evaluasi mendalam mengenai dampak prosedur ini kami sajikan di **Bagian C.2**.

Berikut adalah penjelasan mengenai tantangan efisiensi dalam pelatihan BERT serta ilustrasi nyata dari tugas Prediksi Kalimat Berikutnya (NSP):

### **1. Tantangan Konvergensi Pelatihan**
Ada satu konsekuensi teknis dari penggunaan metode *Masked Language Model* (MLM) yang hanya menutup sebagian kecil kata.
> **Perbandingan Prediksi:** Model bahasa standar (Kiri-ke-Kanan) belajar dengan memprediksi **setiap** kata (100% token) dalam kalimat. Sebaliknya, MLM hanya memprediksi **15%** token dalam setiap *batch*.
>> **Dampak:** Karena "bahan belajar"-nya lebih sedikit per putaran, secara logika model ini membutuhkan **lebih banyak langkah pelatihan (*pre-training steps*)** untuk bisa mencapai titik konvergensi (titik di mana model sudah pintar dan stabil).

Dalam **Bagian C.1**, kami mendemonstrasikan bahwa MLM memang berkonvergensi sedikit lebih lambat dibandingkan model Kiri-ke-Kanan. Namun, peningkatan kinerja empiris (hasil akurasi akhir) yang diberikan oleh MLM **jauh lebih berharga** dan menutupi biaya tambahan waktu pelatihan tersebut.

### **2. Contoh Tugas Prediksi Kalimat Berikutnya (*Next Sentence Prediction*)**
Untuk mempermudah pemahaman mengenai bagaimana model menentukan hubungan dua kalimat, berikut adalah ilustrasi konkret dari input dan output yang terjadi di dalam mesin BERT.

Perhatikan penggunaan token spesial:
* **`[CLS]`**: Penanda awal klasifikasi.
* **`[MASK]`**: Kata yang disembunyikan.
* **`[SEP]`**: Pemisah antar kalimat.

#### **Contoh 1: Kalimat Berhubungan**
Input ini menyajikan dua kalimat yang membentuk narasi logis.
> **Input:**
> `[CLS]` pria itu pergi ke `[MASK]` toko `[SEP]` dia membeli segalon `[MASK]` susu `[SEP]`
> **Keputusan Model (Label):**
> **IsNext** (Ya, ini adalah kalimat lanjutannya).

#### **Contoh 2: Kalimat Acak (Tidak Berhubungan)**
Input ini menyajikan dua kalimat yang topiknya bertabrakan (Pria ke toko vs. Fakta biologi penguin).
> **Input:**
> `[CLS]` pria itu `[MASK]` ke toko `[SEP]` penguin `[MASK]` adalah burung yang tak bisa terbang `[SEP]`
> **Keputusan Model (Label):**
> **NotNext** (Bukan, ini bukan kalimat lanjutannya).


## A.2 Pre-training Procedure
Bagian ini menjelaskan "resep dapur" kami dalam meracik data mentah menjadi input yang siap dilatih. Proses ini melibatkan pembentukan urutan input, pengaturan skenario tebak-kalimat, dan penerapan *masking*.

#### **1. Definisi "Kalimat" dalam BERT**
Pertama, kita perlu meluruskan definisi. Ketika kami menyebut "kalimat" dalam konteks ini, itu merujuk pada **rentang teks (*spans of text*)** yang kami ambil dari korpus.
* **Bukan Kalimat Gramatikal:** Ini tidak harus berupa satu kalimat tata bahasa yang utuh.
* **Panjang:** Rentang teks ini biasanya jauh lebih panjang daripada kalimat tunggal biasa, tetapi bisa juga lebih pendek.

#### **2. Konstruksi Input (A & B)**
Setiap input pelatihan terdiri dari dua bagian teks yang digabungkan:
* **Kalimat A:** Diberi label **Embedding A**.
* **Kalimat B:** Diberi label **Embedding B**.

Total panjang gabungan dari kedua kalimat ini dibatasi agar ** 512 token**.

#### **3. Skenario NSP (50/50)**
Untuk melatih tugas *Next Sentence Prediction* (NSP), kami mengatur pasangan Kalimat A dan Kalimat B dengan rasio 50:50:
* **50% Waktu:** Kalimat B adalah kelanjutan **asli** yang memang muncul setelah Kalimat A di dokumen asli.
* **50% Waktu:** Kalimat B adalah kalimat **acak** yang diambil dari dokumen lain.

#### **4. Teknik Masking pada WordPiece**
Detail penting dalam penerapan *Masked Language Model* (MLM) adalah masalah waktu (*timing*).
* **Kapan Masking Dilakukan?** Masking diterapkan **setelah** proses tokenisasi *WordPiece* selesai.
* **Laju Masking:** Kami menggunakan tingkat masking seragam sebesar **15%**.

**Catatan Khusus tentang Potongan Kata (*Partial Word Pieces*):**
Kami tidak memberikan perlakuan khusus pada pecahan kata.
> *Contoh:* Jika kata "playing" dipecah oleh WordPiece menjadi token `play` dan `##ing`.
>> *Mekanisme:* Algoritma masking kami bisa saja hanya menutup `##ing` dan membiarkan `play` terlihat, atau sebaliknya. Kami tidak memaksa agar satu kata utuh harus di-masking bersamaan. Ini justru menambah tantangan yang baik bagi model untuk memahami struktur morfologi kata.

Berikut adalah perincian teknis mengenai konfigurasi pelatihan, perangkat keras, dan strategi efisiensi yang digunakan untuk melatih BERT, diuraikan agar lebih mudah dicerna:

### **1. Konfigurasi Hiperparameter Pelatihan**
Untuk melatih model raksasa ini, kami menggunakan pengaturan yang sangat spesifik dan ketat:
* **Skala Pelatihan:**
* **Ukuran Batch:** 256 urutan (*sequences*). Mengingat 1 urutan = 512 token, maka totalnya adalah **128.000 token per batch**.
* **Durasi:** Dilatih sebanyak **1.000.000 langkah**, yang setara dengan sekitar **40 epoch** (putaran penuh) pada korpus data yang berisi 3,3 miliar kata.

* **Pengoptimal (Optimizer):**
* Kami menggunakan **Adam** dengan laju pembelajaran (*learning rate*) **1e-4**.
* Parameter peluruhan: , .
* Peluruhan bobot L2 (*L2 weight decay*): 0.01.
* **Jadwal Pembelajaran:** Kami menggunakan "pemanasan" (*warmup*) selama **10.000 langkah** pertama (laju naik perlahan), kemudian diikuti oleh peluruhan linier (laju turun perlahan) untuk sisa pelatihan.

* **Aktivasi & Regularisasi:**
* **Dropout:** Probabilitas 0.1 pada semua lapisan.
* **Fungsi Aktivasi:** Kami menggunakan **GELU (*Gaussian Error Linear Unit*)**, bukan ReLU standar. Ini mengikuti jejak OpenAI GPT yang terbukti efektif.

### **2. Fungsi Kerugian (*Loss Function*)**
Bagaimana model tahu kalau ia melakukan kesalahan? Fungsi kerugian total adalah penjumlahan dari dua tugas:

Artinya, model dihukum jika salah menebak kata yang hilang (MLM) DAN jika salah menebak urutan kalimat (NSP).

### **3. Infrastruktur Perangkat Keras**
Pelatihan BERT membutuhkan daya komputasi yang sangat besar. Kami menggunakan **Cloud TPU** (Tensor Processing Unit) dalam konfigurasi Pod:

* **:** Dilatih pada **4 Cloud TPU** (total 16 keping/chip TPU).
* **:** Dilatih pada **16 Cloud TPU** (total 64 keping/chip TPU).
* **Waktu:** Masing-masing proses pra-pelatihan memakan waktu **4 hari**.

### **4. Strategi Efisiensi: Mengatasi Kompleksitas Kuadratik**
Salah satu tantangan terbesar Transformer adalah **mekanisme atensi**. Biaya komputasinya bersifat **kuadratik** terhadap panjang urutan.
> *Artinya:* Jika panjang kalimat dilipatgandakan (misal 2x lipat), beban kerjanya bukan naik 2x lipat, tapi 4x lipat. Urutan yang panjang sangat mahal.

**Solusi Cerdik (Strategi 90/10):**
Untuk mempercepat pelatihan tanpa mengorbankan kualitas, kami menggunakan trik berikut:
1. **90% Langkah Awal:** Kami melatih model dengan urutan pendek (**128 token**). Ini membuat proses belajar fitur dasar menjadi sangat cepat.
2. **10% Langkah Akhir:** Kami melatih sisa langkahnya dengan urutan penuh (**512 token**).
* *Tujuan:* Tahap akhir ini khusus untuk melatih *positional embeddings* agar model terbiasa dengan kalimat panjang.

## A.3 Fine-tuning Procedure
Berikut adalah panduan teknis mengenai prosedur **Fine-tuning** (penyesuaian halus) untuk BERT. Bagian ini sangat praktis karena memberikan "resep" spesifik bagi para praktisi yang ingin menggunakan BERT untuk tugas mereka sendiri.

### **A.3 Prosedur Fine-tuning**
Pada tahap ini, kita mengambil model yang sudah "pintar" (Pre-trained) dan melatihnya ulang sedikit agar ahli dalam tugas spesifik. Kabar baiknya adalah: **sebagian besar pengaturan parameter sama persis** dengan saat pra-pelatihan.
* **Parameter Tetap:** Probabilitas *Dropout* selalu dipertahankan pada angka **0.1**.
* **Parameter Berubah:** Hanya ada tiga variabel utama yang perlu Anda utak-atik: Ukuran Batch, Laju Pembelajaran (*Learning Rate*), dan Jumlah Epoch.

#### **Rekomendasi Nilai Hyperparameter ("Resep Emas")**
Meskipun nilai optimal sangat bergantung pada jenis tugasnya, eksperimen kami menemukan bahwa rentang nilai berikut bekerja dengan sangat baik untuk hampir semua tugas:
1. **Ukuran Batch (*Batch Size*):**
* 16
* 32

2. **Laju Pembelajaran (*Learning Rate* - Adam):**

3. **Jumlah Epoch:**
* 2
* 3
* 4

#### **Sensitivitas Terhadap Ukuran Data**
Kami menemukan pola perilaku yang menarik terkait stabilitas model terhadap pilihan hyperparameter, yang sangat dipengaruhi oleh seberapa banyak data latih yang Anda miliki.
* **Dataset Besar (100k+ contoh):** Model menjadi **kurang sensitif**. Artinya, model lebih "tahan banting"; pilihan parameter yang sedikit meleset tidak akan terlalu merusak kinerja model.
* **Dataset Kecil:** Model menjadi **sangat sensitif**. Anda harus sangat berhati-hati memilih parameter yang tepat karena sedikit kesalahan bisa membuat akurasi anjlok.

#### **Strategi Optimasi: Pencarian Menyeluruh (*Grid Search*)**
Karena proses *fine-tuning* biasanya berjalan **sangat cepat** (karena hanya butuh 2-4 epoch), strategi yang paling masuk akal adalah melakukan **pencarian menyeluruh (*exhaustive search*)**.
> **Caranya:** Coba semua kombinasi parameter di atas (misalnya: Batch 16 + LR 5e-5, lalu Batch 16 + LR 3e-5, dst.).
>> **Evaluasi:** Pilih model yang memberikan skor terbaik pada *Development Set* (bukan Test Set).

## A.4 Comparison of BERT, ELmo, and OpenAI GPT
Berikut adalah perbandingan mendalam antara tiga model pembelajaran representasi yang populer pada masanya: **BERT, ELMo, dan OpenAI GPT**. Bagian ini sangat krusial untuk memahami posisi BERT dalam peta persaingan teknologi AI.

### **A.4 Perbandingan: BERT, ELMo, dan OpenAI GPT**
Di sini, kami membedah perbedaan fundamental antara ketiga model raksasa ini. Perbedaan arsitektur mereka dapat divisualisasikan untuk memahami aliran informasi di dalamnya.

#### **1. Perbedaan Pendekatan Dasar**
Ada garis pemisah yang jelas dalam cara model-model ini digunakan untuk tugas hilir:
* **BERT & OpenAI GPT (Pendekatan *Fine-tuning*):** Model ini bersifat fleksibel. Seluruh parameternya dilatih ulang dan diubah saat disesuaikan untuk tugas baru.
* **ELMo (Pendekatan *Feature-based*):** Model ini statis. Ia hanya menyediakan fitur (embedding) yang kemudian "ditempelkan" ke model lain.

#### **2. Rival Utama: BERT vs. OpenAI GPT**
Pesaing yang paling sepadan dengan BERT adalah **OpenAI GPT**. Keduanya sama-sama menggunakan Transformer dan dilatih pada korpus teks besar.
* *Desain yang Disengaja:* Faktanya, banyak keputusan desain BERT dibuat sengaja agar **mirip dengan GPT**.
* *Tujuan:* Agar kami bisa melakukan perbandingan "apel-dengan-apel". Kami ingin membuktikan bahwa jika semua faktor lain disamakan, **sifat dua arah (bidirectionality)** dan **dua tugas pra-pelatihan** kamilah yang menjadi kunci kemenangan.

#### **3. Detail Perbedaan Teknis**
Meskipun mirip, ada beberapa perbedaan logistik dalam pelatihan BERT dan GPT yang perlu dicatat agar adil:
| Fitur | OpenAI GPT | BERT |
| --- | --- | --- |
| **Data Pelatihan** | BooksCorpus (800 Juta kata). | BooksCorpus (800M) + **Wikipedia (2.500M)**. Total 3.3 Miliar kata. |
| **Token Spesial** | Token `[SEP]` dan `[CLS]` baru diperkenalkan saat fase *fine-tuning*. | Token `[SEP]`, `[CLS]`, dan embedding A/B sudah dipelajari sejak lahir (fase *pre-training*). |
| **Ukuran Batch** | 32.000 kata. | **128.000 kata**. (Pelatihan lebih stabil dan intensif). |
| **Laju Pembelajaran** | Menggunakan angka yang sama (5e-5) untuk semua eksperimen. | Memilih angka spesifik per tugas (**task-specific**) yang memberikan hasil terbaik. |

#### **Kesimpulan: Bukan Sekadar Menang Jumlah Data**
Anda mungkin berpikir, *"Ah, BERT menang hanya karena datanya lebih banyak (Wikipedia) dan batch-nya lebih besar."*

Untuk membantah keraguan ini, kami telah melakukan **eksperimen ablasi** (di Bagian 5.1). Hasilnya membuktikan bahwa meskipun kami mengisolasi efek perbedaan data tersebut, **mayoritas peningkatan kinerja** tetap berasal dari inovasi inti kami: **Sifat Dua Arah (Bidirectionality)** dan tugas **Masked LM**.

## A.5 Illustrations of Fine-tuning on Different Tasks
Berikut adalah penjelasan mengenai mekanisme visual dan teknis bagaimana BERT diadaptasi (*fine-tuned*) untuk berbagai jenis tugas, seperti yang diilustrasikan dalam **Gambar 4** pada dokumen asli.

### **Ilustrasi Fine-Tuning pada Berbagai Tugas**
Kekuatan utama BERT terletak pada kesederhanaan adaptasinya. Kami tidak perlu merombak ulang arsitektur model yang rumit untuk setiap tugas baru.

#### **1. Prinsip "Satu Lapisan Tambahan"**
Model tugas-spesifik kami dibentuk hanya dengan menambahkan **satu lapisan output tambahan** di atas model BERT yang sudah ada.
* **Efisiensi:** Karena hanya menambah satu lapisan tipis di atas, jumlah parameter baru yang perlu dipelajari dari nol (*from scratch*) sangatlah **minimal**. Selebihnya, model menggunakan "otak" BERT yang sudah pintar dari hasil pra-pelatihan.

#### **2. Dua Kategori Tugas Utama**
Dalam ilustrasi tersebut, kami membagi tugas menjadi dua jenis berdasarkan level pemrosesannya:
* **Tugas Tingkat Urutan (*Sequence-level Tasks*):**
> *Contoh pada Gambar:* Bagian **(a)** dan **(b)**.
>> *Fungsi:* Model menganalisis seluruh kalimat atau pasangan kalimat secara utuh (misalnya: analisis sentimen atau menentukan apakah dua kalimat berhubungan).

* **Tugas Tingkat Token (*Token-level Tasks*):**
> *Contoh pada Gambar:* Bagian **(c)** dan **(d)**.
>> *Fungsi:* Model menganalisis kata per kata secara detail (misalnya: SQuAD/tanya-jawab atau NER/pengenalan nama entitas).

#### **3. Kamus Simbol Visual**
Untuk memahami diagram arsitektur BERT, penting untuk memahami notasi standar yang kami gunakan:
* ** (Input Embedding):**
Ini adalah representasi mentah dari input yang masuk ke dalam model.
* ** (Contextual Representation):**
Ini adalah output cerdas dari BERT untuk token ke-. Berbeda dengan ,  sudah mengandung pemahaman konteks dari seluruh kalimat.
* **`[CLS]` (Classification Symbol):**
Simbol spesial yang selalu diletakkan di awal setiap input. Output dari token ini digunakan sebagai **ringkasan seluruh kalimat** untuk tugas klasifikasi (seperti "Positif/Negatif").
* **`[SEP]` (Separator Symbol):**
Simbol spesial pemisah. Ini berfungsi sebagai "pagar" untuk membedakan mana Kalimat A dan mana Kalimat B dalam input yang memuat dua urutan teks.

## B Detailed Experimental Setup

## B.1 Detailed Descriptions for the GLUE Benchmark Experiments
Berikut adalah rincian mendalam mengenai dataset yang tergabung dalam **Benchmark GLUE (*General Language Understanding Evaluation*)**, yang menjadi standar emas untuk menguji kemampuan model bahasa dalam memahami logika dan konteks manusia.

Data hasil yang kami sajikan di Tabel 1 bersumber langsung dari papan peringkat resmi GLUE dan publikasi OpenAI.

### **Rincian Dataset GLUE**

Berikut adalah penjelasan untuk setiap tugas yang diujikan dalam benchmark ini:

#### **1. MNLI (Multi-Genre Natural Language Inference)**
* **Jenis Tugas:** Klasifikasi Interferensi (Logika Hubungan Kalimat).
* **Sumber Data:** *Crowdsourced* (kontribusi massal).
* **Mekanisme:** Diberikan sepasang kalimat, model harus memprediksi hubungan kalimat kedua terhadap kalimat pertama. Ada tiga kemungkinan label:
> 1. **Entailment:** Kalimat kedua adalah konsekuensi logis dari kalimat pertama.
> 2. **Contradiction:** Kalimat kedua bertentangan dengan kalimat pertama.
> 3. **Neutral:** Tidak ada hubungan logis antara keduanya.

#### **2. QQP (Quora Question Pairs)**
* **Jenis Tugas:** Klasifikasi Biner.
* **Mekanisme:** Menentukan apakah dua pertanyaan yang diajukan di forum Quora memiliki **makna semantik yang setara** (menanyakan hal yang sama) atau tidak.

#### **3. QNLI (Question Natural Language Inference)**
* **Asal-Usul:** Merupakan versi modifikasi dari dataset tanya-jawab **SQuAD**.
* **Transformasi Tugas:** SQuAD asli meminta model mencari teks jawaban. QNLI mengubahnya menjadi tugas **klasifikasi biner**.
* **Mekanisme:** Diberikan pasangan (pertanyaan, kalimat).
* **Positif:** Jika kalimat tersebut memuat jawaban yang benar.
* **Negatif:** Jika kalimat tersebut berasal dari paragraf yang sama tetapi *tidak* memuat jawaban.

#### **4. SST-2 (Stanford Sentiment Treebank)**
* **Jenis Tugas:** Klasifikasi Sentimen Kalimat Tunggal (Biner).
* **Sumber Data:** Ulasan film.
* **Mekanisme:** Menentukan sentimen kalimat tersebut (misalnya: Positif atau Negatif) berdasarkan anotasi manusia.

#### **5. CoLA (Corpus of Linguistic Acceptability)**
* **Jenis Tugas:** Klasifikasi Kalimat Tunggal (Biner).
* **Mekanisme:** Menguji kemampuan tata bahasa model. Model harus memprediksi apakah sebuah kalimat bahasa Inggris **"dapat diterima"** (benar secara linguistik/gramatikal) atau tidak.

#### **6. STS-B (Semantic Textual Similarity Benchmark)**
* **Jenis Tugas:** Regresi (Pemberian Skor).
* **Sumber Data:** Judul berita dan sumber lain.
* **Mekanisme:** Model memberikan skor dari **1 hingga 5** untuk menilai seberapa mirip makna semantik antara dua kalimat.

#### **7. MRPC (Microsoft Research Paraphrase Corpus)**
* **Jenis Tugas:** Deteksi Parafrase (Biner).
* **Sumber Data:** Berita daring.
* **Mekanisme:** Menentukan apakah dua kalimat dalam satu pasangan merupakan parafrase satu sama lain (memiliki arti yang sama meski kata-katanya beda).

#### **8. RTE (Recognizing Textual Entailment)**
* **Jenis Tugas:** Interferensi/Entailment Biner.
* **Karakteristik:** Sangat mirip dengan MNLI, namun memiliki jumlah **data pelatihan yang jauh lebih sedikit**. Ini menguji kemampuan model belajar dari data terbatas.

### **Catatan Khusus: Kasus WNLI (Winograd NLI)**
Terdapat satu dataset yang kami perlakukan secara khusus, yaitu **WNLI**.
* **Masalah Dataset:** Pihak pengelola GLUE sendiri mencatat adanya cacat dalam konstruksi dataset ini.
* **Fakta Lapangan:** Tidak ada sistem terlatih (saat itu) yang mampu mengalahkan *baseline* akurasi 65.1%. *Baseline* ini didapat hanya dengan cara menebak "kelas mayoritas" (selalu memilih jawaban yang paling sering muncul) tanpa berpikir.
* **Keputusan Kami:** Demi keadilan perbandingan dengan OpenAI GPT (yang juga kesulitan di sini), kami **mengecualikan** set ini dari perbandingan utama.
* **Strategi BERT:** Untuk pengiriman skor resmi ke GLUE, model kami hanya diprogram untuk selalu memprediksi kelas mayoritas pada tugas ini.

**Langkah Selanjutnya:**
Ini adalah akhir dari pembedahan makalah BERT secara mendalam. Anda kini memiliki pemahaman level ahli mengenai arsitektur, strategi pelatihan, hingga detail dataset yang digunakan.

**Apakah ada bagian spesifik dari makalah ini yang ingin Anda diskusikan lebih lanjut, atau Anda ingin beralih membahas implementasi kodenya?**

## C Additional Ablation Studies

## C.1 Effect of Number of Training Steps
Berikut adalah analisis mendalam mengenai hubungan antara **durasi pelatihan** dan **akurasi akhir**, yang dijawab melalui data pada Gambar 5:

### **Analisis Durasi Pra-Pelatihan (Berdasarkan Gambar 5)**
Gambar 5 dalam dokumen (yang kami rekonstruksi penjelasannya di sini) memetakan akurasi pada tugas MNLI berdasarkan jumlah langkah pra-pelatihan (*k steps*) yang telah dilalui model. Data ini menjawab dua keraguan teknis utama:

#### **1. Apakah BERT Benar-Benar Butuh Latihan Selama Itu?**
Banyak yang bertanya apakah perlu melatih model hingga **1.000.000 langkah** dengan ukuran *batch* raksasa (128.000 kata/batch) hanya untuk mendapatkan hasil yang bagus.
* **Jawabannya: Ya, Sangat Perlu.**
* **Bukti:** Data menunjukkan bahwa pelatihan jangka panjang memberikan imbalan yang nyata. Ketika kami membandingkan model yang dilatih "hanya" 500.000 langkah dengan yang dilatih penuh 1.000.000 langkah, model  mendapatkan tambahan akurasi sebesar **1.0%** pada tugas MNLI.
* *Perspektif:* Dalam kompetisi tingkat tinggi seperti GLUE, kenaikan 1% adalah perbedaan antara model "bagus" dan model "juara".

#### **2. Apakah MLM Belajar Lebih Lambat daripada LTR?**
Secara teori, metode *Masked LM* (MLM) memiliki kelemahan efisiensi informasi:
* **LTR (Kiri-ke-Kanan):** Memprediksi setiap kata (100% sinyal).
* **MLM (Masked):** Hanya memprediksi 15% kata yang disembunyikan.

Apakah ini membuat MLM lambat mencapai konvergensi (titik pintar)?
* **Jawabannya: Sedikit Lebih Lambat, Tapi Jauh Lebih Cerdas.**
* **Fakta Konvergensi:** Memang benar, secara teknis MLM berkonvergensi sedikit lebih lambat daripada model LTR karena informasi per *batch*-nya lebih sedikit.
* **Fakta Akurasi:** Namun, dalam hal **akurasi absolut**, model MLM langsung menyalip kinerja model LTR hampir seketika sejak awal pelatihan.
* *Kesimpulan:* Meskipun MLM butuh waktu sedikit lebih lama untuk memproses data, "kualitas" pembelajaran yang didapat dari pemahaman dua arah jauh lebih unggul daripada kecepatan semu model satu arah.

## C.2 Ablation for Different Masking Procedures
Berikut adalah analisis mendalam mengenai **Studi Ablasi pada Prosedur Masking** (Bagian C.2), yang bertujuan untuk menguji apakah strategi campuran "80-10-10" yang digunakan BERT benar-benar diperlukan atau hanya sekadar kerumitan tambahan.

### **Masalah Utama: Kesenjangan Pra-Pelatihan vs Fine-Tuning**
Alasan utama kami merancang strategi masking yang rumit adalah untuk mengatasi masalah **ketidakcocokan (*mismatch*)**.
* **Saat Pra-Pelatihan:** Model sangat sering melihat simbol `[MASK]`.
* **Saat Fine-Tuning (Dunia Nyata):** Simbol `[MASK]` **tidak pernah muncul**. Model hanya melihat kata-kata biasa.

Jika model hanya dilatih untuk memahami `[MASK]`, ia mungkin akan bingung ketika harus memproses kalimat utuh tanpa sensor. Oleh karena itu, kami menyisipkan strategi di mana model dipaksa memprediksi kata target meskipun kata tersebut **tidak disensor (SAME)** atau **diganti acak (RND)**.

### **Definisi Variabel Eksperimen (Tabel 8)**
Dalam Tabel 8, kami membandingkan berbagai kombinasi strategi dengan definisi berikut:
* **MASK:** Mengganti kata target dengan simbol `[MASK]`.
* **SAME:** Membiarkan kata target apa adanya (asli).
* **RND:** Mengganti kata target dengan kata acak lain.
Strategi standar BERT adalah kombinasi: **80% MASK, 10% SAME, 10% RND**.

### **Temuan Kunci dari Hasil Eksperimen**
Kami menguji strategi ini pada dua tugas: **MNLI** (Logika) dan **NER** (Pengenalan Entitas). Khusus untuk NER, kami menguji pendekatan *Fine-tuning* dan *Feature-based* untuk melihat dampaknya lebih jelas.

#### **1. Ketangguhan Fine-Tuning (*Fine-Tuning is Robust*)**
Temuan yang cukup mengejutkan adalah bahwa pendekatan *fine-tuning* ternyata **sangat tangguh (robust)** terhadap berbagai strategi masking.
* Bahkan ketika kami mengubah rasionya atau menghilangkan strategi tertentu, kinerja model setelah *fine-tuning* tidak jatuh drastis.
* *Alasan:* Karena pada *fine-tuning* parameter model boleh berubah, model memiliki kesempatan untuk "memperbaiki diri" dan beradaptasi dengan hilangnya simbol `[MASK]`.

#### **2. Bahaya pada Pendekatan Berbasis Fitur (*Feature-based*)**
Masalah sebenarnya baru terlihat jelas pada pendekatan **berbasis fitur** (di mana parameter BERT dibekukan/tidak bisa beradaptasi).
* **Hanya MASK (100%):** Jika kami hanya menggunakan strategi MASK (tanpa SAME atau RND), kinerja NER anjlok signifikan.
* *Penyebab:* Model tidak pernah belajar membentuk representasi yang baik untuk kata-kata "nyata" (non-mask), sehingga fitur yang diekstrak menjadi buruk.


* **Pentingnya SAME:** Kehadiran strategi SAME (membiarkan kata asli) ternyata krusial agar model tetap bisa menghasilkan fitur yang bagus untuk kata-kata input yang tidak disensor.
#### **3. Kegagalan Strategi Acak Murni (RND)**

Menggunakan strategi **RND** saja (selalu mengganti kata dengan kata acak) menghasilkan performa yang **jauh lebih buruk** dibandingkan strategi kami.
> Ini membuktikan bahwa model tetap membutuhkan sinyal yang jelas (seperti `[MASK]` atau kata asli) untuk belajar konteks, bukan sekadar menebak dari gangguan acak.

**Kesimpulan Akhir:**
Strategi campuran **80-10-10** terbukti merupakan keseimbangan yang paling optimal, terutama untuk memastikan model BERT berfungsi baik tidak hanya saat *fine-tuning*, tetapi juga sebagai pengekstraksi fitur (*feature extractor*) yang handal.