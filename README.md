# Eksperimen Preprocessing Data - Mamikos Promo Price Prediction

Repositori ini berisi tahapan Eksplorasi Data (EDA) dan otomatisasi pipeline preprocessing data kost promo Mamikos di berbagai kota besar di Indonesia. Proyek ini dibangun sebagai bagian dari Kriteria 1 Proyek Akhir kelas **Membangun Sistem Machine Learning** (Dicoding).

## Anggota Tim / Siswa
- **Nama**: dzakwanalifi

## Struktur Repositori
```text
Eksperimen_SML_dzakwanalifi
├── preprocessing/                       # Folder preprocessing
├── mamikos_promo_ngebut_indonesia.csv   # Dataset mentah (raw) hasil scraping (2.061 data)
├── mamikos_preprocessing_dataset.csv    # Dataset hasil preprocessing siap latih
├── Eksperimen_dzakwanalifi.ipynb # Jupyter Notebook (EDA & Preprocessing)
└── automate_dzakwanalifi.py   # Script otomatisasi preprocessing
```

## Deskripsi Tahapan Preprocessing
1. **Pembersihan Harga**:
   - Menghapus format teks Rupiah (`Rp`) dan pemisah ribuan (`.`) pada kolom `original_price` dan `discount_price`.
   - Mengonversi tipe data harga dari string ke float/numerik.
2. **Penanganan Missing Values**:
   - Imputasi nilai kosong (`NaN`) pada kolom `rating` menggunakan nilai median (`4.8`).
   - Imputasi nilai kosong pada kolom `original_price` menggunakan nilai dari `discount_price` (apabila tidak ada promo harga coret).
3. **Ekstraksi Fasilitas (One-Hot Encoding)**:
   - Mengekstrak 6 fasilitas utama dari teks gabungan di kolom `facilities`: `Kasur`, `WiFi`, `Kloset Duduk`, `K. Mandi Dalam`, `AC`, dan `Akses 24 Jam`.
4. **Encoding Variabel Kategorikal**:
   - Menerapkan *Label Encoding* pada variabel `city`, `gender`, dan `district`.
5. **Feature Scaling & Train-Test Split**:
   - Membagi data menjadi 80% data latih (train) dan 20% data uji (test).
   - Melakukan scaling fitur numerik menggunakan `StandardScaler`.
