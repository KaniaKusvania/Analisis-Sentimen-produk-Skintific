# 💄 Analisis Sentimen Ulasan Moisturizer Skintific Menggunakan Machine Learning

[cite_start]Proyek ini berfokus pada pengembangan sistem **analisis sentimen otomatis** untuk memahami dan mengukur **persepsi serta kepuasan pelanggan** terhadap produk *moisturizer* dari *brand* **Skintific** berdasarkan ulasan yang dikumpulkan dari platform **Female Daily**[cite: 1].

***
## 🎯 Business Objective

Penelitian ini bertujuan untuk:
* [cite_start]**Mengembangkan sistem** analisis sentimen berbasis algoritma *Machine Learning* untuk menganalisis ulasan produk secara otomatis[cite: 8].
* [cite_start]**Membantu perusahaan (Skintific)** memahami persepsi dan tingkat kepuasan pelanggan secara *real-time*[cite: 8].
* [cite_start]**Mengidentifikasi area peningkatan produk** dan mengoptimalkan strategi *marketing*[cite: 8].
* [cite_start]**Menyediakan calon konsumen** informasi yang lebih objektif dan berbasis data (*data-driven*) untuk pengambilan keputusan pembelian[cite: 8].

***
## 🛠️ Peta Jalan Proyek: Dari Teks Kasar ke Prediksi Model

### 1. 📂 Data & Sumber

| Keterangan | Detail |
| :--- | :--- |
| **Sumber Data** | [cite_start]Ulasan produk *moisturizer* Skintific dari platform Female Daily[cite: 16]. |
| **Target** | [cite_start]Mengklasifikasikan sentimen ulasan ke dalam 4 kategori: **Positif, Negatif, Netral,** dan **Other**[cite: 20]. |
| **Ukuran Data** | **1838 entri** ulasan. |

### 2. 📊 Exploratory Data Analysis (EDA)

#### Tipe Kulit 
* Dataset ulasan didominasi oleh pengguna dengan tipe kulit **Neutral** ($\text{1202}$) dan **Combination** ($\text{750}$).
* [cite_start]Tingkat frekuensi yang **tidak merata (*imbalanced*)** antar kategori tipe kulit dicatat sebagai karakteristik dataset[cite: 33].

#### Kata Kunci Ulasan (*Word Cloud*) 

[Image of the Word Cloud of Comments]

* [cite_start]Kata-kata yang paling dominan adalah “aku”, “dan”, “juga”, “tapi”, dan “karena” (kata umum)[cite: 38].
* [cite_start]Kata kunci yang sering muncul dan relevan dengan produk meliputi: **“skin barrier”, “moisturizer”, “produk”, “kulit”, “bagus”, “cocok”**, dan **“jerawat”**[cite: 39].

#### Distribusi Sentimen 

[Image of the Sentiment Distribution Bar Chart]

* Sentimen **Neutral** ($\text{867}$) adalah kategori yang paling dominan, diikuti oleh sentimen **Positive** ($\text{669}$).
* Sentimen **Negative** ($\text{294}$) jumlahnya jauh lebih sedikit.
* [cite_start]Secara keseluruhan, persepsi pengguna cenderung **netral hingga positif**[cite: 46].

***
## 🔧 Data Pre-processing & Feature Engineering

Data mentah melalui serangkaian tahapan pembersihan dan transformasi untuk dipersiapkan bagi model *Machine Learning*.

### A. Tahapan Pre-processing

| Tahap | Keterangan |
| :--- | :--- |
| **1. Data Cleaning** | [cite_start]Penghapusan angka, tanda baca, simbol, emoji, dan spasi berlebihan menggunakan *regular expression* untuk mengurangi *noise*[cite: 56]. |
| **2. Case Folding** | [cite_start]Konversi seluruh teks menjadi huruf kecil (*lowercase*) untuk memastikan konsistensi[cite: 63]. |
| **3. Tokenization** | [cite_start]Pemecahan kalimat menjadi unit-unit kata (token)[cite: 68]. |
| **4. Stopword Removal** | [cite_start]Penghapusan kata-kata umum bahasa Indonesia yang tidak signifikan (seperti 'dan', 'yang') menggunakan daftar dari NLTK[cite: 78]. |
| **5. Label Encoding** | Konversi label sentimen teks (**Negative, Neutral, Other, Positive**) menjadi representasi numerik ($\text{0, 1, 2, 3}$) untuk input model. |

### B. Metode Ekstraksi Fitur Teks

| Metode | Deskripsi | Fitur $\text{max\_features}$ |
| :--- | :--- | :--- |
| **TF-IDF** | [cite_start]Merepresentasikan kata berdasarkan frekuensi kemunculannya (TF) dan seberapa unik kata tersebut di seluruh dokumen (IDF)[cite: 101]. | [cite_start]$\text{5000}$ [cite: 101] |
| **Bag-of-Words (BoW)** | [cite_start]Merepresentasikan teks sebagai vektor frekuensi hitungan (*count frequency*) setiap kata[cite: 86]. | [cite_start]$\text{5000}$ [cite: 89] |
| **N-grams** | [cite_start]Mengatasi kelemahan BoW dengan mempertimbangkan urutan kata (unigram dan bigram) untuk menangkap konteks[cite: 123, 128]. | [cite_start]$\text{5000}$ [cite: 129] |
| **Word Embeddings (Word2Vec)** | [cite_start]Mengubah kata menjadi vektor 100-dimensi (CBOW), di mana vektor dokumen dihasilkan dari rata-rata vektor kata[cite: 115, 118]. | $\text{100}$ |

***
## 🤖 Model Machine Learning yang Digunakan

Proyek ini mengeksplorasi dan mengevaluasi kinerja tiga model utama, menggunakan kombinasi fitur **N-gram + Seleksi Fitur $\chi^2$ ($\text{k}=50$)**.

| Model | Akurasi | F1-Score (W. Avg) | Metrik Kunci & *Best Params* |
| :--- | :--- | :--- | :--- |
| **SVC (Tuned)** | 0.8940 | 0.89 | **Neutral** *Recall* 1.00; **Positive** *Precision* 0.99. $\text{Params}: \text{'C': 1, 'kernel': 'linear'}$. |
| **Random Forest (Tuned)** | 0.8940 | 0.89 | **Neutral** *Recall* 1.00; **Positive** *Precision* 0.97. $\text{Params}: \text{'max\_depth': 20, 'min\_samples\_leaf': 2, 'min\_samples\_split': 2, 'n\_estimators': 100}$. |
| **MLP (Deep Learning)** | **0.8995** | 0.89 | Kinerja sedikit unggul, *Recall* 1.00 pada **Neutral**. | N/A |
| **SVC + IndoBERT (Transfer Learning)** | [cite_start]0.7228% [cite: 202] | N/A | [cite_start]Akurasi lebih rendah dari model berbasis *term-frequency*[cite: 204]. | N/A |

***
## 🔍 Analisis Kesalahan Utama (*Error Analysis*)

* [cite_start]**Bias Sentimen Kuat:** Semua model menunjukkan kecenderungan kuat untuk mengklasifikasikan sampel **Positive** yang sebenarnya sebagai **Neutral** (terjadi 30-31 kasus pada setiap model)[cite: 198].
* [cite_start]**Kegagalan Kelas Minoritas:** Semua model **gagal total** mengklasifikasikan kelas **Other** ($\text{F1-score}$ 0.00) karena jumlah sampel (*support*) yang sangat minim (hanya $\text{2}$ sampel)[cite: 190, 196, 200].

***
## ➡️ Kesimpulan Aksi

Model **Tuned SVC** dan **Random Forest** adalah solusi paling direkomendasikan karena memberikan akurasi sangat tinggi ($\sim 89.4\%$) dengan kompleksitas implementasi yang lebih rendah. [cite_start]Namun, keberlanjutan proyek harus fokus pada penanganan *class imbalance* untuk meningkatkan validitas prediksi pada sentimen minoritas[cite: 200].
