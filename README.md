# 💄 Analisis Sentimen Ulasan Moisturizer Skintific Menggunakan Machine Learning

Proyek ini berfokus pada pengembangan sistem **analisis sentimen otomatis** untuk memahami dan mengukur **persepsi serta kepuasan pelanggan** terhadap produk *moisturizer* dari *brand* **Skintific** berdasarkan ulasan yang dikumpulkan dari platform **Female Daily**.

***
## 🎯 Business Objective

Penelitian ini bertujuan untuk:
* **Mengembangkan sistem** analisis sentimen berbasis algoritma *Machine Learning* untuk menganalisis ulasan produk secara otomatis.
* **Membantu perusahaan (Skintific)** memahami persepsi dan tingkat kepuasan pelanggan secara *real-time*.
* **Mengidentifikasi area peningkatan produk** dan mengoptimalkan strategi *marketing*.
* **Menyediakan calon konsumen** informasi yang lebih objektif dan berbasis data (*data-driven*) untuk pengambilan keputusan pembelian.

***
## 🛠️ Peta Jalan Proyek: Dari Teks Kasar ke Prediksi Model

### A. 📂 Data & Sumber

| Keterangan | Detail |
| :--- | :--- |
| **Sumber Data** | Ulasan produk *moisturizer* Skintific dari platform Female Daily. |
| **Target** | Mengklasifikasikan sentimen ulasan ke dalam 4 kategori: **Positif, Negatif, Netral,** dan **Other**. |
| **Ukuran Data** | **1838 entri** ulasan. |

### B. 📊 Exploratory Data Analysis (EDA)

#### 1. Tipe Kulit 
* Dataset ulasan didominasi oleh pengguna dengan tipe kulit **Neutral** ($\text{1202}$) dan **Combination** ($\text{750}$).
* Tingkat frekuensi yang **tidak merata (*imbalanced*)** antar kategori tipe kulit dicatat sebagai karakteristik dataset.

#### 2. Kata Kunci Ulasan (*Word Cloud*) 
* Kata-kata yang paling dominan adalah “aku”, “dan”, “juga”, “tapi”, dan “karena” (kata umum).
* Kata kunci yang sering muncul dan relevan dengan produk meliputi: **“skin barrier”, “moisturizer”, “produk”, “kulit”, “bagus”, “cocok”**, dan **“jerawat”**.

#### 3. Distribusi Sentimen 
* Sentimen **Neutral** ($\text{867}$) adalah kategori yang paling dominan, diikuti oleh sentimen **Positive** ($\text{669}$).
* Sentimen **Negative** ($\text{294}$) jumlahnya jauh lebih sedikit.
* Secara keseluruhan, persepsi pengguna cenderung **netral hingga positif**.

***
## 🔧 Data Pre-processing & Feature Engineering

Data mentah melalui serangkaian tahapan pembersihan dan transformasi untuk dipersiapkan bagi model *Machine Learning*.

### A. Tahapan Pre-processing

| Tahap | Keterangan |
| :--- | :--- |
| **1. Data Cleaning** | Penghapusan angka, tanda baca, simbol, emoji, dan spasi berlebihan menggunakan *regular expression* untuk mengurangi *noise*. |
| **2. Case Folding** | Konversi seluruh teks menjadi huruf kecil (*lowercase*) untuk memastikan konsistensi. |
| **3. Tokenization** | Pemecahan kalimat menjadi unit-unit kata (token). |
| **4. Stopword Removal** | Penghapusan kata-kata umum bahasa Indonesia yang tidak signifikan (seperti 'dan', 'yang') menggunakan daftar dari NLTK. |
| **5. Label Encoding** | Konversi label sentimen teks (**Negative, Neutral, Other, Positive**) menjadi representasi numerik ($\text{0, 1, 2, 3}$) untuk input model. |

### B. Metode Ekstraksi Fitur Teks

| Metode | Deskripsi | Fitur $\text{max\_features}$ |
| :--- | :--- | :--- |
| **TF-IDF** | Merepresentasikan kata berdasarkan frekuensi kemunculannya (TF) dan seberapa unik kata tersebut di seluruh dokumen (IDF). | $\text{5000}$ |
| **Bag-of-Words (BoW)** | Merepresentasikan teks sebagai vektor frekuensi hitungan (*count frequency*) setiap kata. | $\text{5000}$ |
| **N-grams** | Mengatasi kelemahan BoW dengan mempertimbangkan urutan kata (unigram dan bigram) untuk menangkap konteks. | $\text{5000}$ |
| **Word Embeddings (Word2Vec)** | Mengubah kata menjadi vektor 100-dimensi (CBOW), di mana vektor dokumen dihasilkan dari rata-rata vektor kata. | $\text{100}$ |

***
## 🤖 Model Machine Learning yang Digunakan

Proyek ini mengeksplorasi dan mengevaluasi kinerja tiga model utama, menggunakan kombinasi fitur **N-gram + Seleksi Fitur $\chi^2$ ($\text{k}=50$)**.

| Model | Akurasi | F1-Score (W. Avg) | Metrik Kunci & *Best Params* |
| :--- | :--- | :--- | :--- |
| **SVC (Tuned)** | 0.8940 | 0.89 | **Neutral** *Recall* 1.00; **Positive** *Precision* 0.99. $\text{Params}: \text{'C': 1, 'kernel': 'linear'}$. |
| **Random Forest (Tuned)** | 0.8940 | 0.89 | **Neutral** *Recall* 1.00; **Positive** *Precision* 0.97. $\text{Params}: \text{'max\_depth': 20, 'min\_samples\_leaf': 2, 'min\_samples\_split': 2, 'n\_estimators': 100}$. |
| **MLP (Deep Learning)** | **0.8995** | 0.89 | Kinerja sedikit unggul, *Recall* 1.00 pada **Neutral**. | N/A |
| **SVC + IndoBERT (Transfer Learning)** | 0.7228% | N/A | Akurasi lebih rendah dari model berbasis *term-frequency*. | N/A |

***
## 🔍 Analisis Kesalahan Utama (*Error Analysis*)

* **Bias Sentimen Kuat:** Semua model menunjukkan kecenderungan kuat untuk mengklasifikasikan sampel **Positive** yang sebenarnya sebagai **Neutral** (terjadi 30-31 kasus pada setiap model).
* **Kegagalan Kelas Minoritas:** Semua model **gagal total** mengklasifikasikan kelas **Other** ($\text{F1-score}$ 0.00) karena jumlah sampel (*support*) yang sangat minim (hanya $\text{2}$ sampel).

***
## ➡️ Kesimpulan Aksi

Model **Tuned SVC** dan **Random Forest** adalah solusi paling direkomendasikan karena memberikan akurasi sangat tinggi ($\sim 89.4\%$) dengan kompleksitas implementasi yang lebih rendah. Namun, keberlanjutan proyek harus fokus pada penanganan *class imbalance* untuk meningkatkan validitas prediksi pada sentimen minoritas.
