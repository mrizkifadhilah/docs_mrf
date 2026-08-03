# Attention is all you need

Ashish Vaswani∗
Google Brain
avaswani@google.com

Llion Jones∗
Google Research
llion@google.com

Noam Shazeer∗
Google Brain
noam@google.com

Aidan N. Gomez∗†
University of Toronto
aidan@cs.toronto.edu

Niki Parmar∗
Google Research
nikip@google.com

Jakob Uszkoreit∗
Google Research
usz@google.com

Łukasz Kaiser∗
Google Brain
lukaszkaiser@google.com

Illia Polosukhin∗‡
illia.polosukhin@gmail.com

## Abstract

Model transduksi urutan yang dominan didasarkan pada *recurrent neural networks* atau *convolutional neural networks* yang kompleks yang mencakup *encoder* dan *decoder*. Model-model dengan kinerja terbaik juga menghubungkan *encoder* dan *decoder* melalui mekanisme atensi. Kami mengusulkan arsitektur jaringan baru yang sederhana, Transformer, yang didasarkan semata-mata pada mekanisme atensi, meniadakan rekurensi dan konvolusi sepenuhnya. Eksperimen pada dua tugas penerjemahan mesin menunjukkan bahwa model-model ini unggul dalam kualitas sekaligus lebih dapat diparalelkan dan membutuhkan waktu pelatihan yang jauh lebih singkat. Model kami mencapai 28,4 BLEU pada tugas penerjemahan Bahasa Inggris-ke-Jerman WMT 2014, meningkat lebih dari 2 BLEU dibandingkan hasil terbaik yang ada, termasuk *ensemble*. Pada tugas penerjemahan Bahasa Inggris-ke-Prancis WMT 2014, model kami menetapkan skor BLEU *state-of-the-art* model tunggal yang baru sebesar 41,8 setelah pelatihan selama 3,5 hari menggunakan delapan GPU, sebagian kecil dari biaya pelatihan model-model terbaik dari literatur. Kami menunjukkan bahwa Transformer memiliki kemampuan generalisasi yang baik terhadap tugas-tugas lain dengan menerapkannya secara sukses pada *constituency parsing* Bahasa Inggris, baik dengan data pelatihan yang besar maupun terbatas.

## 1 - Introduction

*Recurrent neural networks* (RNN), khususnya *long short-term memory* [*LSTM*, Hochreiter et al, 1997] dan *gated recurrent neural networks* [*GRU*, Chung et al, 2014] , telah dikukuhkan sebagai pendekatan *state-of-the-art* — standar terkini yang paling mutakhir — dalam masalah pemodelan dan transduksi urutan, seperti pemodelan bahasa dan penerjemahan mesin [*Seq2Seq*, Sutskever et al, 2014; *Neural Machine Translation*, Bahdanau et al, 2014; *RNN Encoder-Decoder*, Cho et al, 2014]. Berbagai upaya sejak saat itu terus dilakukan untuk memperluas batasan model bahasa rekuren dan arsitektur *encoder-decoder* [*GNMT*, Wu et al, 2016; *Attention-based NMT*, Luong et al, 2015; *Language Modeling Limits*, Jozefowicz et al, 2016].

Model-model rekuren biasanya memfaktorkan komputasi di sepanjang posisi simbol dari urutan input dan output. Dengan menyelaraskan posisi ke langkah-langkah dalam waktu komputasi, model-model tersebut menghasilkan urutan status tersembunyi $h_{t}$ sebagai fungsi dari status tersembunyi sebelumnya $h_{t-1}$ dan input untuk posisi $t$. Sifat sekuensial yang inheren ini menghalangi paralelisasi dalam contoh pelatihan, yang menjadi krusial pada panjang urutan yang lebih besar, karena batasan memori membatasi pembuatan batch di seluruh contoh. Penelitian terbaru telah mencapai peningkatan signifikan dalam efisiensi komputasi melalui teknik faktorisasi [Factorization Tricks, Kuchaiev et al, 2017] dan komputasi kondisional [MoE, Shazeer et al, 2017] , sembari juga meningkatkan performa model pada kasus terakhir. Namun, batasan mendasar dari komputasi sekuensial tetap ada.

Mekanisme atensi telah menjadi bagian integral dari model pemodelan dan transduksi urutan yang kompeten dalam berbagai tugas, memungkinkan pemodelan dependensi tanpa memperhatikan jaraknya dalam urutan input atau output [*Joint Learning*, Bahdanau et al, 2014; *Structured Attention*, Kim et al, 2017]. Namun, dalam hampir semua kasus kecuali beberapa pengecualian [*Decomposable Attention*, Parikh et al, 2016] , mekanisme atensi tersebut digunakan bersama dengan jaringan rekuren.

Dalam karya ini, kami mengusulkan *Transformer*, sebuah arsitektur model yang menghindari rekurensi dan sebagai gantinya bergantung sepenuhnya pada mekanisme atensi untuk menarik dependensi global antara input dan output. *Transformer* memungkinkan paralelisasi yang secara signifikan lebih besar dan dapat mencapai standar *state-of-the-art* baru dalam kualitas terjemahan setelah dilatih hanya selama dua belas jam menggunakan delapan GPU P100.

## 2 - Background

Tujuan untuk mengurangi komputasi sekuensial juga menjadi landasan bagi [*Extended Neural GPU*, Kaiser et al, 2016], [*ByteNet*, Kalchbrenner et al, 2017], dan [*ConvS2S*, Gehring et al, 2017], yang semuanya menggunakan *convolutional neural networks* — jaringan saraf konvensional — sebagai blok pembangun dasar, guna menghitung representasi tersembunyi secara paralel untuk seluruh posisi input dan output. Dalam model-model ini, jumlah operasi yang diperlukan untuk menghubungkan sinyal dari dua posisi input atau output arbitrer meningkat seiring bertambahnya jarak antar posisi, yakni secara linear untuk [*ConvS2S*, Gehring et al, 2017] dan logaritmik untuk [*ByteNet*, Kalchbrenner et al, 2017]. Hal ini mempersulit proses pembelajaran dependensi antara posisi-posisi yang berjauhan [*Gradient Flow*, Hochreiter et al, 2001]. Dalam *Transformer*, hal ini direduksi menjadi jumlah operasi konstan, meskipun dengan konsekuensi berkurangnya resolusi efektif akibat rata-rata posisi tertimbang atensi (*attention-weighted positions*), suatu efek yang kami tanggulangi dengan *Multi-Head Attention* sebagaimana dijelaskan pada bagian 3.2.

*Self-attention* — terkadang disebut *intra-attention* — adalah mekanisme atensi yang menghubungkan posisi-posisi berbeda dari satu urutan tunggal untuk menghitung representasi dari urutan tersebut. *Self-attention* telah berhasil digunakan dalam berbagai tugas termasuk pemahaman bacaan (*reading comprehension*), peringkasan abstraktif (*abstractive summarization*), keterpautan tekstual (*textual entailment*), dan pembelajaran representasi kalimat yang tidak bergantung pada tugas tertentu [*LSTM-networks*, Cheng et al, 2016; *Decomposable Attention*, Parikh et al, 2016; *Deep Reinforced Model*, Paulus et al, 2017; *Self-attentive Sentence Embedding*, Lin et al, 2017].

*End-to-end memory networks* didasarkan pada mekanisme atensi rekuren alih-alih rekurensi yang selaras dengan urutan (*sequence-aligned recurrence*) dan telah terbukti berperforma baik pada tugas menjawab pertanyaan bahasa sederhana serta pemodelan bahasa [*End-to-end Memory Networks*, Sukhbaatar et al, 2015].

Namun, sejauh pengetahuan kami, *Transformer* adalah model transduksi pertama yang bergantung sepenuhnya pada *self-attention* untuk menghitung representasi input dan outputnya tanpa menggunakan *RNN* atau konvolusi yang selaras dengan urutan. Pada bagian selanjutnya, kami akan mendeskripsikan *Transformer*, memberikan motivasi di balik penggunaan *self-attention*, serta membahas keunggulannya dibandingkan model-model seperti [*Neural GPUs*, Kaiser et al, 2016], [*ByteNet*, Kalchbrenner et al, 2017], dan [*ConvS2S*, Gehring et al, 2017].

## 3 - Model Architecture

Sebagian besar model transduksi urutan saraf (neural sequence transduction models) yang kompetitif memiliki struktur encoder-decoder [RNN Encoder-Decoder, Cho et al, 2014; Neural Machine Translation, Bahdanau et al, 2014; Seq2Seq, Sutskever et al, 2014]. Di sini, encoder memetakan urutan input dari representasi simbol $(x_1, \dots, x_n)$ ke sebuah urutan representasi kontinu $z = (z_1, \dots, z_n)$. Berdasarkan $z$, decoder kemudian menghasilkan urutan output $(y_1, \dots, y_m)$ dari simbol-simbol tersebut, satu elemen pada satu waktu. Pada setiap langkah, model bersifat auto-regressive — regresif diri — [PixelCNN, Van den Oord et al, 2016], yang mengonsumsi simbol-simbol yang dihasilkan sebelumnya sebagai input tambahan saat menghasilkan simbol berikutnya.

Transformer mengikuti arsitektur keseluruhan ini dengan menggunakan tumpukan self-attention dan lapisan yang terhubung sepenuhnya secara point-wise (point-wise, fully connected layers) baik untuk encoder maupun decoder, yang masing-masing ditunjukkan pada bagian kiri dan kanan Gambar 1.

## 3.1 - Encoder and Decoder Stacks

**Encoder**: Encoder terdiri dari sebuah tumpukan $N = 6$ lapisan identik. Setiap lapisan memiliki dua sub-lapisan. Sub-lapisan pertama adalah mekanisme multi-head self-attention, dan yang kedua adalah position-wise fully connected feed-forward network — jaringan saraf umpan maju yang terhubung sepenuhnya pada setiap posisi — yang sederhana. Kami menerapkan residual connection — koneksi sisa — [ResNet, He et al, 2016] di sekitar masing-masing dari kedua sub-lapisan tersebut, yang diikuti oleh layer normalization — normalisasi lapisan — [Layer Normalization, Ba et al, 2016]. Dengan demikian, output dari setiap sub-lapisan adalah $\text{LayerNorm}(x + \text{Sublayer}(x))$, di mana $\text{Sublayer}(x)$ adalah fungsi yang diimplementasikan oleh sub-lapisan itu sendiri. Untuk memfasilitasi residual connection ini, semua sub-lapisan dalam model, serta lapisan embedding, menghasilkan output dengan dimensi $d_{\text{model}} = 512$.

**Decoder**: Decoder juga terdiri dari sebuah tumpukan N=6 lapisan identik. Sebagai tambahan dari dua sub-lapisan yang ada pada setiap lapisan encoder, decoder menyisipkan sub-lapisan ketiga, yang menjalankan multi-head attention di atas output dari tumpukan encoder. Serupa dengan encoder, kami menerapkan residual connection di sekitar masing-masing sub-lapisan, yang diikuti oleh layer normalization. Kami juga memodifikasi sub-lapisan self-attention pada tumpukan decoder untuk mencegah suatu posisi memperhatikan (attending to) posisi-posisi setelahnya. Mekanisme masking — penyelubungan — ini, dikombinasikan dengan fakta bahwa output embeddings digeser satu posisi, memastikan bahwa prediksi untuk posisi i hanya dapat bergantung pada output yang telah diketahui pada posisi yang lebih kecil dari i.

## 3.2 - Attention

Fungsi atensi dapat dideskripsikan sebagai pemetaan sebuah query dan sekumpulan pasangan key-value (kunci-nilai) ke sebuah output, di mana query, keys, values, dan output semuanya merupakan vektor. Output dihitung sebagai jumlah tertimbang (weighted sum) dari values, di mana bobot yang diberikan pada setiap value dihitung melalui fungsi kompatibilitas dari query dengan key yang bersesuaian.

## 3.2.1 - Scaled Dot-Product Attention

Kami menyebut mekanisme atensi khusus kami sebagai "Scaled Dot-Product Attention" (Gambar 2). Input terdiri dari queries dan keys berdimensi $d_{k}$, serta values berdimensi $d_{v}$. Kami menghitung hasil kali titik (dot products) dari query dengan seluruh keys, membagi masing-masing dengan $\sqrt{d_{k}}$, dan menerapkan fungsi softmax untuk mendapatkan bobot pada values.

Dalam praktiknya, kami menghitung fungsi atensi pada sekumpulan queries secara simultan, yang dikemas bersama ke dalam matriks $Q$. Keys dan values juga dikemas bersama ke dalam matriks $K$ dan $V$. Kami menghitung matriks output sebagai:

Dalam praktiknya, kami menghitung fungsi atensi pada sekumpulan queries secara simultan, yang dikemas bersama ke dalam matriks $Q$. Keys dan values juga dikemas bersama ke dalam matriks $K$ dan $V$. Kami menghitung matriks output sebagai:

$$
Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V
$$

Dua fungsi atensi yang paling umum digunakan adalah additive attention [Neural Machine Translation, Bahdanau et al, 2014] dan dot-product (multiplicative) attention. Dot-product attention identik dengan algoritma kami, kecuali untuk faktor skala $\frac{1}{\sqrt{d_{k}}}$. Additive attention menghitung fungsi kompatibilitas menggunakan feed-forward network dengan satu lapisan tersembunyi. Meskipun keduanya memiliki kompleksitas teoritis yang serupa, dot-product attention jauh lebih cepat dan lebih efisien dalam penggunaan ruang secara praktis, karena dapat diimplementasikan menggunakan kode perkalian matriks yang sangat teroptimasi.

Meskipun untuk nilai $d_{k}$ yang kecil kedua mekanisme tersebut berperforma serupa, additive attention mengungguli dot product attention tanpa penskalaan untuk nilai $d_{k}$ yang lebih besar [Massive Exploration of NMT, Britz et al, 2017]. Kami menduga bahwa untuk nilai $d_{k}$ yang besar, besaran dot products meningkat secara signifikan, sehingga mendorong fungsi softmax ke wilayah di mana ia memiliki gradien yang sangat kecil (untuk mengilustrasikan mengapa dot products membesar, asumsikan bahwa komponen dari $q$ dan $k$ adalah variabel acak independen dengan rata-rata 0 dan varians 1; maka hasil kali titik mereka, $q\cdot k=\sum_{i=1}^{d_{k}}q_{i}k_{i}$, memiliki rata-rata 0 dan varians $d_{k}$). Untuk menanggulangi efek ini, kami menskala dot products dengan $\frac{1}{\sqrt{d_{k}}}$.

## 3.2.2 - Multi-Head Attention

Alih-alih menjalankan fungsi atensi tunggal dengan keys, values, dan queries berdimensi $d_{model}$, kami menemukan bahwa memproyeksikan queries, keys, dan values secara linear sebanyak $h$ kali dengan proyeksi linear berbeda yang dipelajari masing-masing ke dimensi $d_k$, $d_k$, dan $d_v$, memberikan hasil yang bermanfaat. Pada setiap versi proyeksi dari queries, keys, dan values ini, kami kemudian menjalankan fungsi atensi secara paralel, yang menghasilkan nilai output berdimensi $d_v$. Nilai-nilai ini digabungkan (concatenated) dan sekali lagi diproyeksikan, menghasilkan nilai akhir sebagaimana diilustrasikan pada Gambar 2.

Multi-head attention memungkinkan model untuk secara bersama-sama memperhatikan informasi dari subruang representasi (representation subspaces) yang berbeda pada posisi yang berbeda pula. Dengan attention head tunggal, proses perataan (averaging) menghambat hal tersebut.

$$
MultiHead(Q, K, V) = Concat(head_1, \dots, head_h)W^O
$$

$$
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

Di mana proyeksi merupakan matriks parameter $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, dan $W^O \in \mathbb{R}^{hd_v \times d_{model}}$.

Dalam karya ini, kami menggunakan $h = 8$ lapisan atensi paralel, atau heads. Untuk masing-masing lapisan tersebut, kami menggunakan $d_k = d_v = d_{model}/h = 64$. Karena pengurangan dimensi pada setiap head, total biaya komputasi serupa dengan single-head attention dengan dimensionalitas penuh.


## 3.2.3 - Applications of Attention in our Model

Transformer menggunakan multi-head attention dalam tiga cara yang berbeda:

* Dalam lapisan "encoder-decoder attention", queries berasal dari lapisan decoder sebelumnya, sementara memory keys dan values berasal dari output encoder. Hal ini memungkinkan setiap posisi dalam decoder untuk memperhatikan seluruh posisi dalam urutan input. Mekanisme ini meniru mekanisme atensi encoder-decoder tipikal dalam model sequence-to-sequence seperti [GNMT, Wu et al, 2016; Neural Machine Translation, Bahdanau et al, 2014; ConvS2S, Gehring et al, 2017].

* Encoder mengandung lapisan self-attention. Dalam sebuah lapisan self-attention, seluruh keys, values, dan queries berasal dari tempat yang sama, dalam hal ini, output dari lapisan sebelumnya pada encoder. Setiap posisi dalam encoder dapat memperhatikan seluruh posisi pada lapisan sebelumnya di encoder tersebut.
* Demikian pula, lapisan self-attention dalam decoder memungkinkan setiap posisi dalam decoder untuk memperhatikan seluruh posisi dalam decoder hingga dan termasuk posisi tersebut. Kami perlu mencegah aliran informasi ke arah kiri (leftward information flow) dalam decoder untuk mempertahankan sifat auto-regressive — regresif diri. Kami mengimplementasikan hal ini di dalam scaled dot-product attention dengan melakukan masking out — menutupi atau menyetel ke $-\infty$ — seluruh nilai pada input softmax yang bersesuaian dengan koneksi terlarang. Lihat Gambar 2.

## 3.3 - Position-wise Feed-Forward Networks

Selain sub-lapisan atensi, setiap lapisan pada encoder dan decoder kami memiliki sebuah fully connected feed-forward network — jaringan saraf umpan maju yang terhubung sepenuhnya — yang diterapkan pada setiap posisi secara terpisah dan identik. Jaringan ini terdiri dari dua transformasi linear dengan aktivasi ReLU di antaranya:

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Meskipun transformasi linear tersebut sama di berbagai posisi yang berbeda, mereka menggunakan parameter yang berbeda dari satu lapisan ke lapisan lainnya. Cara lain untuk mendeskripsikan hal ini adalah sebagai dua konvolusi dengan ukuran kernel 1. Dimensionalitas input dan output adalah $d_{\text{model}} = 512$, dan lapisan internalnya memiliki dimensionalitas $d_{ff} = 2048$.

## 3.4 - Embedding and Softmax

Serupa dengan model transduksi urutan lainnya, kami menggunakan learned embeddings — penyematan yang dipelajari — untuk mengubah token input dan token output menjadi vektor berdimensi $d_{\text{model}}$. Kami juga menggunakan transformasi linear terpelajar yang lazim dan fungsi softmax untuk mengubah output decoder menjadi prediksi probabilitas token berikutnya. Dalam model kami, kami berbagi matriks bobot yang sama di antara kedua lapisan embedding dan transformasi linear pra-softmax, serupa dengan [Using Output Embedding, Press et al, 2016]. Pada lapisan embedding, kami mengalikan bobot tersebut dengan $\sqrt{d_{\text{model}}}$.

## 3.5 - Positional Encoding

Karena model kami tidak mengandung rekurensi maupun konvolusi, agar model dapat memanfaatkan urutan sekuensial, kami harus menginjeksikan beberapa informasi mengenai posisi relatif atau absolut dari token dalam urutan tersebut. Untuk tujuan ini, kami menambahkan "positional encodings" (pengodean posisi) pada input embeddings di bagian bawah tumpukan encoder dan decoder. Positional encodings memiliki dimensi $d_{model}$ yang sama dengan embeddings, sehingga keduanya dapat dijumlahkan. Terdapat banyak pilihan positional encodings, baik yang dipelajari (learned) maupun yang tetap (fixed) [ConvS2S, Gehring et al, 2017].

Dalam karya ini, kami menggunakan fungsi sinus dan kosinus dengan frekuensi yang berbeda:

$$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$$

$$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$$

di mana $pos$ adalah posisi dan $i$ adalah dimensi. Artinya, setiap dimensi dari positional encoding bersesuaian dengan sebuah sinusoid. Panjang gelombangnya membentuk progresi geometris dari $2\pi$ hingga $10000 \cdot 2\pi$. Kami memilih fungsi ini karena kami berhipotesis bahwa hal tersebut akan memungkinkan model untuk belajar memperhatikan posisi relatif dengan mudah, karena untuk setiap offset tetap $k$, $PE_{pos+k}$ dapat direpresentasikan sebagai fungsi linear dari $PE_{pos}$.

Kami juga bereksperimen dengan menggunakan positional embeddings yang dipelajari [ConvS2S, Gehring et al, 2017] sebagai gantinya, dan menemukan bahwa kedua versi tersebut menghasilkan hasil yang hampir identik (lihat Tabel 3 baris (E)). Kami memilih versi sinusoidal karena hal tersebut memungkinkan model untuk melakukan ekstrapolasi ke panjang urutan yang lebih panjang daripada yang ditemui selama pelatihan.

## 4 - Why Self-Attention

Pada bagian ini, kami membandingkan berbagai aspek lapisan self-attention dengan lapisan rekuren dan konvolusional yang umum digunakan untuk memetakan satu urutan representasi simbol dengan panjang variabel $(x_1, \dots, x_n)$ ke urutan lain dengan panjang yang sama $(z_1, \dots, z_n)$, dengan $x_i, z_i \in \mathbb{R}^d$, seperti lapisan tersembunyi pada encoder atau decoder transduksi urutan tipikal. Sebagai motivasi atas penggunaan self-attention, kami mempertimbangkan tiga kriteria utama (desiderata).

Pertama adalah total kompleksitas komputasi per lapisan. Kedua adalah jumlah komputasi yang dapat diparalelkan, yang diukur dengan jumlah minimum operasi sekuensial yang diperlukan.

Kriteria ketiga adalah panjang jalur antara dependensi jarak jauh (long-range dependencies) di dalam jaringan. Mempelajari dependensi jarak jauh merupakan tantangan utama dalam banyak tugas transduksi urutan. Salah satu faktor kunci yang memengaruhi kemampuan untuk mempelajari dependensi tersebut adalah panjang jalur yang harus dilalui oleh sinyal maju dan mundur di dalam jaringan. Semakin pendek jalur ini di antara kombinasi posisi apa pun dalam urutan input dan output, semakin mudah untuk mempelajari dependensi jarak jauh [Gradient Flow, Hochreiter et al, 2001] . Oleh karena itu, kami juga membandingkan panjang jalur maksimum antara dua posisi input dan output mana pun dalam jaringan yang terdiri dari jenis lapisan yang berbeda.

Sebagaimana dicatat dalam Tabel 1, lapisan self-attention menghubungkan semua posisi dengan jumlah operasi eksekusi sekuensial yang konstan, sedangkan lapisan rekuren memerlukan $O(n)$ operasi sekuensial. Dari segi kompleksitas komputasi, lapisan self-attention lebih cepat daripada lapisan rekuren ketika panjang urutan $n$ lebih kecil daripada dimensionalitas representasi $d$, yang merupakan kasus paling umum pada representasi kalimat yang digunakan oleh model-model state-of-the-art dalam penerjemahan mesin, seperti representasi word-piece [GNMT, Wu et al, 2016] dan byte-pair [Neural Machine Translation of Rare Words, Sennrich et al, 2015]. Untuk meningkatkan performa komputasi pada tugas-tugas yang melibatkan urutan yang sangat panjang, self-attention dapat dibatasi hanya untuk mempertimbangkan lingkungan berukuran $r$ dalam urutan input yang berpusat di sekitar posisi output masing-masing. Hal ini akan meningkatkan panjang jalur maksimum menjadi $O(n/r)$. Kami berencana untuk menyelidiki pendekatan ini lebih lanjut dalam pekerjaan di masa depan.

Satu lapisan konvolusional dengan lebar kernel $k < n$ tidak menghubungkan semua pasangan posisi input dan output. Untuk melakukannya, diperlukan tumpukan sebanyak $O(n/k)$ lapisan konvolusional dalam kasus kernel berdampingan (contiguous kernels), atau $O(\log_k(n))$ dalam kasus dilated convolutions [ByteNet, Kalchbrenner et al, 2017], yang meningkatkan panjang jalur terlama antara dua posisi mana pun dalam jaringan. Lapisan konvolusional umumnya lebih mahal daripada lapisan rekuren sebesar faktor $k$. Namun, separable convolutions — konvolusi terpisah — [Xception, Chollet, 2016] menurunkan kompleksitas secara signifikan menjadi $O(k \cdot n \cdot d + n \cdot d^2)$. Meskipun demikian, bahkan dengan $k = n$, kompleksitas separable convolution setara dengan kombinasi lapisan self-attention dan lapisan point-wise feed-forward, yang merupakan pendekatan yang kami ambil dalam model kami.

Sebagai manfaat tambahan, self-attention dapat menghasilkan model yang lebih mudah diinterpretasikan. Kami memeriksa distribusi atensi dari model kami dan menyajikan serta mendiskusikan contoh-contohnya dalam lampiran. Tidak hanya masing-masing attention head yang jelas belajar untuk melakukan tugas yang berbeda, banyak di antaranya tampaknya menunjukkan perilaku yang berkaitan dengan struktur sintaksis dan semantik dari kalimat.

## 5 - Training

## 5.1 - Training Data and Batching

Kami melatih model menggunakan kumpulan data standar WMT 2014 Inggris-Jerman yang terdiri dari sekitar 4,5 juta pasangan kalimat. Kalimat-kalimat tersebut dikodekan menggunakan byte-pair encoding [Massive Exploration of NMT, Britz et al, 2017], yang memiliki kosakata sumber-target bersama sekitar 37.000 token. Untuk pasangan Inggris-Prancis, kami menggunakan kumpulan data WMT 2014 Inggris-Prancis yang jauh lebih besar, terdiri dari 36 juta kalimat dan membagi token ke dalam kosakata word-piece sebanyak 32.000 unit [GNMT, Wu et al, 2016]. Pasangan kalimat dikelompokkan dalam batch berdasarkan perkiraan panjang urutan. Setiap batch pelatihan berisi sekumpulan pasangan kalimat yang mencakup kurang lebih 25.000 token sumber dan 25.000 token target.

## 5.2 - Hardware and Schedule

Kami melatih model kami pada satu mesin dengan 8 GPU NVIDIA P100. Untuk model basis (base models) kami yang menggunakan hiperparameter yang telah dijelaskan di seluruh makalah ini, setiap langkah pelatihan memakan waktu sekitar 0,4 detik. Kami melatih model basis tersebut selama total 100.000 langkah atau 12 jam. Untuk model besar (big models) kami (sebagaimana dideskripsikan pada baris bawah Tabel 3), waktu setiap langkah adalah 1,0 detik. Model-model besar tersebut dilatih selama 300.000 langkah atau setara dengan 3,5 hari.

## 5.3 - Optimizer

Kami menggunakan pengoptimal Adam [Adam, Kingma et al, 2014] dengan $\beta_1 = 0,9$, $\beta_2 = 0,98$, dan $\epsilon = 10^{-9}$. Kami memvariasikan laju pembelajaran (learning rate) selama masa pelatihan sesuai dengan rumus berikut:

$$lrate = d_{model}^{-0,5} \cdot \min(step\_num^{-0,5}, step\_num \cdot warmup\_steps^{-1,5})$$

Hal ini bersesuaian dengan peningkatan laju pembelajaran secara linear untuk $warmup\_steps$ langkah pelatihan pertama, dan setelahnya menurun secara proporsional terhadap akar kuadrat terbalik dari nomor langkah. Kami menggunakan $warmup\_steps = 4000$.

## 5.4 - Regularization

Kami menerapkan tiga jenis regularisasi selama pelatihan:

**Residual Dropout**: Kami menerapkan dropout [Dropout, Srivastava et al, 2014] pada output dari setiap sub-lapisan, sebelum ditambahkan ke input sub-lapisan dan dinormalisasi . Selain itu, kami menerapkan dropout pada hasil penjumlahan embeddings dan positional encodings baik pada tumpukan encoder maupun decoder. Untuk model basis (base model), kami menggunakan laju $P_{drop} = 0,1$.

**Label Smoothing**: Selama pelatihan, kami menerapkan label smoothing dengan nilai $\epsilon_{ls} = 0,1$ [Inception Architecture, Szegedy et al, 2015] . Hal ini memperburuk perplexity, karena model belajar untuk menjadi lebih tidak yakin (more unsure), namun meningkatkan akurasi dan skor BLEU.

## 6 - Results

## 6.1 - Machine Translation

Pada tugas penerjemahan Bahasa Inggris-ke-Jerman WMT 2014, model transformer besar — [Transformer (big), Vaswani et al, 2017] — mengungguli model-model terbaik yang dilaporkan sebelumnya (termasuk model ensemble) dengan selisih lebih dari 2,0 BLEU, serta menetapkan skor BLEU state-of-the-art baru sebesar 28,4. Konfigurasi model ini tercantum pada baris terakhir Tabel 3. Proses pelatihan memakan waktu 3,5 hari menggunakan 8 GPU P100. Bahkan model basis kami melampaui seluruh model tunggal dan ensemble yang dipublikasikan sebelumnya, dengan biaya pelatihan yang jauh lebih kecil dibandingkan model kompetitif lainnya.

Pada tugas penerjemahan Bahasa Inggris-ke-Prancis WMT 2014, model besar kami mencapai skor BLEU sebesar 41,0, mengungguli seluruh model tunggal yang dipublikasikan sebelumnya, dengan biaya pelatihan kurang dari 1/4 dari model state-of-the-art sebelumnya. Model [Transformer (big), Vaswani et al, 2017] yang dilatih untuk Bahasa Inggris-ke-Prancis menggunakan laju dropout $P_{drop} = 0,1$, alih-alih 0,3.

Untuk model-model basis, kami menggunakan model tunggal yang diperoleh dengan merata-ratakan 5 checkpoint terakhir yang dicatat dengan interval 10 menit. Untuk model-model besar, kami merata-ratakan 20 checkpoint terakhir. Kami menggunakan beam search dengan ukuran beam 4 dan penalti panjang $\alpha = 0,6$ [GNMT, Wu et al, 2016]. Hiperparameter ini dipilih setelah eksperimen pada set pengembangan. Kami menetapkan panjang output maksimum selama inferensi menjadi panjang input + 50, namun menghentikan proses lebih awal jika memungkinkan [GNMT, Wu et al, 2016].

Tabel 2 merangkum hasil kami dan membandingkan kualitas terjemahan serta biaya pelatihan dengan arsitektur model lain dari literatur. Kami mengestimasi jumlah operasi titik mengambang (floating point operations) yang digunakan untuk melatih sebuah model dengan mengalikan waktu pelatihan, jumlah GPU yang digunakan, dan estimasi kapasitas titik mengambang presisi tunggal yang berkelanjutan dari setiap GPU.

## 6.2 - Model Variations

Untuk mengevaluasi pentingnya berbagai komponen dari Transformer, kami memvariasikan model basis kami dengan berbagai cara, serta mengukur perubahan performa pada tugas penerjemahan Bahasa Inggris-ke-Jerman menggunakan set pengembangan, newstest2013. Kami menggunakan beam search sebagaimana dijelaskan pada bagian sebelumnya, namun tanpa melakukan perataan checkpoint (checkpoint averaging). Kami menyajikan hasil ini pada Tabel 3.

Pada Tabel 3 baris (A), kami memvariasikan jumlah attention heads serta dimensi key dan value atensi, dengan menjaga jumlah komputasi tetap konstan, sebagaimana dijelaskan pada Bagian 3.2.2. Meskipun single-head attention memberikan hasil 0,9 BLEU lebih buruk daripada pengaturan terbaik, kualitas juga menurun jika jumlah heads terlalu banyak.

Pada Tabel 3 baris (B), kami mengamati bahwa pengurangan ukuran key atensi $d_k$ memperburuk kualitas model. Hal ini menunjukkan bahwa menentukan kompatibilitas tidaklah mudah dan fungsi kompatibilitas yang lebih canggih daripada dot product mungkin bermanfaat. Selanjutnya, kami mengamati pada baris (C) dan (D) bahwa, sesuai ekspektasi, model yang lebih besar memberikan hasil yang lebih baik, dan dropout sangat membantu dalam menghindari over-fitting. Pada baris (E), kami mengganti sinusoidal positional encoding kami dengan learned positional embeddings — penyematan posisi terpelajar — [ConvS2S, Gehring et al, 2017] dan mengamati hasil yang hampir identik dengan model basis.


## 6.3 - English Constituency Parsing

Untuk mengevaluasi apakah Transformer dapat digeneralisasi ke tugas-tugas lain, kami melakukan eksperimen pada English constituency parsing (analisis konstituensi bahasa Inggris). Tugas ini menghadirkan tantangan spesifik: output tunduk pada batasan struktural yang kuat dan secara signifikan lebih panjang daripada input. Selain itu, model RNN sequence-to-sequence belum mampu mencapai hasil state-of-the-art dalam rezim data kecil.

Kami melatih transformer 4-lapisan dengan $d_{\text{model}} = 1024$ pada bagian Wall Street Journal (WSJ) dari Penn Treebank , yaitu sekitar 40 ribu kalimat pelatihan. Kami juga melatihnya dalam pengaturan semi-terawasi (semi-supervised), menggunakan korpus high-confidence dan BerkeleyParser yang lebih besar dengan sekitar 17 juta kalimat. Kami menggunakan kosakata sebanyak 16 ribu token untuk pengaturan khusus WSJ dan kosakata 32 ribu token untuk pengaturan semi-terawasi.

Kami hanya melakukan sejumlah kecil eksperimen untuk memilih dropout (baik atensi maupun residual), laju pembelajaran, dan ukuran beam pada set pengembangan Seksi 22; semua parameter lainnya tetap tidak berubah dari model penerjemahan basis Inggris-ke-Jerman. Selama inferensi, kami meningkatkan panjang output maksimum menjadi panjang input + 300. Kami menggunakan ukuran beam 21 dan $\alpha = 0,3$ baik untuk pengaturan khusus WSJ maupun semi-terawasi.

Hasil kami pada Tabel 4 menunjukkan bahwa meskipun kurangnya penalaan spesifik tugas (task-specific tuning), model kami berperforma sangat baik, menghasilkan hasil yang lebih baik daripada semua model yang dilaporkan sebelumnya kecuali [Recurrent Neural Network Grammar, Dyer et al, 2016]. Berbeda dengan model RNN sequence-to-sequence [Grammar as a Foreign Language, Vinyals & Kaiser et al, 2015], Transformer mengungguli [BerkeleyParser, Petrov et al, 2006] bahkan ketika hanya melatih pada set pelatihan WSJ yang berjumlah 40 ribu kalimat.

## 7 - Conclusion

Dalam karya ini, kami menghadirkan *Transformer*, model transduksi urutan pertama yang sepenuhnya didasarkan pada atensi. Arsitektur ini menggantikan lapisan rekuren yang paling umum digunakan dalam arsitektur *encoder-decoder* dengan *multi-headed self-attention*.

Untuk tugas-tugas penerjemahan, *Transformer* dapat dilatih secara signifikan lebih cepat daripada arsitektur yang didasarkan pada lapisan rekuren atau konvolusional. Baik pada tugas penerjemahan Bahasa Inggris-ke-Jerman WMT 2014 maupun Bahasa Inggris-ke-Prancis WMT 2014, kami mencapai standar *state-of-the-art* yang baru. Pada tugas pertama, model terbaik kami bahkan mengungguli seluruh model *ensemble* yang pernah dilaporkan sebelumnya.

Kami sangat antusias dengan masa depan model berbasis atensi dan berencana untuk menerapkannya pada tugas-tugas lain. Kami berencana memperluas *Transformer* untuk menangani masalah yang melibatkan modalitas input dan output selain teks. Selain itu, kami akan menyelidiki mekanisme atensi lokal yang terbatas agar dapat menangani input dan output besar seperti gambar, audio, dan video secara efisien. Membuat proses generasi menjadi kurang sekuensial juga merupakan salah satu tujuan penelitian kami selanjutnya.

Kode yang kami gunakan untuk melatih dan mengevaluasi model kami tersedia di [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor).

## References

* [*Layer Normalization*, Ba et al, 2016] 
* [*Neural Machine Translation*, Bahdanau et al, 2014] 
* [*Massive Exploration of NMT*, Britz et al, 2017] 
* [*LSTM-networks for Machine Reading*, Cheng et al, 2016] 
* [*RNN Encoder-Decoder*, Cho et al, 2014] 
* [*Xception*, Chollet, 2016] 
* [*Gated Recurrent Neural Networks*, Chung et al, 2014] 
* [*RNN Grammars*, Dyer et al, 2016] 
* [*ConvS2S*, Gehring et al, 2017] 
* [*Generating Sequences with RNN*, Graves, 2013] 
* [*ResNet*, He et al, 2016] 
* [*Gradient Flow in Recurrent Nets*, Hochreiter et al, 2001] 
* [*LSTM*, Hochreiter et al, 1997] 
* [*Self-training PCFG Grammars*, Huang et al, 2009] 
* [*Exploring Limits of Language Modeling*, Jozefowicz et al, 2016] 
* [*Active Memory*, Kaiser et al, 2016] 
* [*Neural GPUs*, Kaiser et al, 2016] 
* [*ByteNet*, Kalchbrenner et al, 2017] 
* [*Structured Attention Networks*, Kim et al, 2017] 
* [*Adam*, Kingma et al, 2014] 
* [*Factorization Tricks*, Kuchaiev et al, 2017] 
* [*Self-attentive Sentence Embedding*, Lin et al, 2017] 
* [*Multi-task Seq2Seq*, Luong et al, 2015] 
* [*Attention-based NMT*, Luong et al, 2015] 
* [*Penn Treebank*, Marcus et al, 1993] 
* [*Effective Self-training for Parsing*, McClosky et al, 2006] 
* [*Decomposable Attention*, Parikh et al, 2016] 
* [*Deep Reinforced Model*, Paulus et al, 2017] 
* [*BerkeleyParser*, Petrov et al, 2006] 
* [*Using Output Embedding*, Press et al, 2016] 
* [*Neural Machine Translation of Rare Words*, Sennrich et al, 2015] 
* [*MoE*, Shazeer et al, 2017] 
* [*Dropout*, Srivastava et al, 2014] 
* [*End-to-end Memory Networks*, Sukhbaatar et al, 2015] 
* [*Seq2Seq*, Sutskever et al, 2014] 
* [*Rethinking Inception Architecture*, Szegedy et al, 2015] 
* [*Grammar as a Foreign Language*, Vinyals et al, 2015] 
* [*GNMT*, Wu et al, 2016] 
* [*Deep Recurrent Models*, Zhou et al, 2016] 
* [*Shift-reduce Constituent Parsing*, Zhu et al, 2013] 

