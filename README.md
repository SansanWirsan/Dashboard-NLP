# 📰 NLP Dashboard — Reuters-21578 Decision Tree

Dashboard Streamlit profesional & interaktif untuk Mini Project NLP klasifikasi
berita Reuters-21578 menggunakan **Decision Tree (CART)** dengan 3 eksperimen
feature extraction: **Bag-of-Words**, **N-gram**, dan **TF-IDF**.

---

## ✨ Fitur

| Halaman | Isi |
|---|---|
| 🏠 Beranda | Ringkasan dataset, distribusi kelas, alur kerja |
| 📊 EDA | Sample data, distribusi train/test, panjang dokumen, top tokens per kelas |
| 🧹 Pre-processing | Visual side-by-side teks asli vs bersih, sandbox teks bebas |
| 🧪 Eksperimen Model | Latih DT dengan BoW / N-gram / TF-IDF, lihat metrik, confusion matrix, feature importance, tree plot |
| 🆚 Perbandingan | Jalankan 3 eksperimen sekaligus, bar chart, radar chart, waktu training |
| 🔍 Prediksi | Uji teks baru dengan model yang sudah dilatih |
| 📚 Panduan | Dokumentasi lengkap |

---

## 🚀 Langkah-Langkah

### 1. Siapkan folder project
```bash
mkdir nlp_dashboard && cd nlp_dashboard
# Salin app.py dan requirements.txt ke folder ini
```

### 2. Buat virtual environment (rekomendasi)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan dashboard
```bash
streamlit run app.py
```
Browser otomatis membuka `http://localhost:8501`.

### 5. Upload dataset
Di **sidebar kiri**, upload:
- `ModApte_train.csv`
- `ModApte_test.csv`

(Atau centang **"Gunakan data demo"** untuk uji cepat tanpa dataset.)

### 6. Eksplorasi & Eksperimen
1. Buka tab **EDA** → pahami distribusi data.
2. Buka tab **Pre-processing** → lihat hasil cleaning.
3. Buka tab **Eksperimen Model** → latih masing-masing metode.
4. Buka tab **Perbandingan** → klik *"Jalankan semua eksperimen"*.
5. Buka tab **Prediksi** → uji berita baru.

### 7. Deploy (opsional)
**Streamlit Cloud** (gratis):
1. Push folder ke GitHub.
2. Buka https://share.streamlit.io → *New app* → pilih repo & `app.py`.
3. Selesai, dapat URL publik.

---

## 📂 Struktur File
```
nlp_dashboard/
├── app.py              # Aplikasi Streamlit
├── requirements.txt    # Dependencies
└── README.md           # Dokumentasi
```

## ⚙️ Parameter Default
- `Top-N kelas` = 10 (earn, acq, crude, trade, money-fx, interest, ship, sugar, coffee, gold)
- `max_features` = 5000
- `max_depth` = 20
- `min_samples_split` = 5

Semua bisa diubah live dari sidebar.

---

## 🧠 Tips Laporan
- Screenshot tab **Perbandingan** (radar + bar chart) untuk bagian *Hasil*.
- Screenshot **Confusion Matrix** & **Feature Importance** dari tab Eksperimen.
- Gunakan tombol **⬇️ Download CSV** di tab Perbandingan untuk lampiran.
