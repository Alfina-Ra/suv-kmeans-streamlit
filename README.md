# K-Means Clustering and Interactive Streamlit Recommendation System for SUV Performance and Efficiency Analysis

Aplikasi web interaktif (Streamlit) untuk mengelompokkan SUV berdasarkan performa dan efisiensi dengan K-Means,
serta merekomendasikan SUV berdasarkan tenaga mesin dan efisiensi bahan bakar minimum.

Interactive Streamlit application that clusters SUVs by performance and efficiency using K-Means and
filters SUVs by minimum horsepower and fuel economy.

## Isi repositori / Contents

| File | Fungsi / Purpose |
|---|---|
| `code.py` | Aplikasi Streamlit / Streamlit application |
| `generate_results.py` | Skrip pembangkit tabel dan gambar hasil untuk naskah / script that generates the result tables and figures |
| `requirements.txt` | Daftar library Python / Python dependencies |

## Dataset

File `SUVs_Ranker_Table.csv` dari dataset publik **SUV Buyers Guide 2023 Specifications** di Kaggle:
https://www.kaggle.com/datasets/ahmedeltom/suv-buyers-guide-2023-specifications

Dataset tidak disertakan di repositori ini. Unduh dari tautan di atas dan ikuti lisensi di halaman Kaggle tersebut.
The dataset is not included in this repository; please download it from the link above and follow its license.

Kolom yang dibutuhkan / required columns: `Hp`, `Torque_Nm`, `Fuel_Eco_km/l`, `0_100`, `Curb_W_kg`
(opsional / optional: `Category`, `Model`).

## Instalasi / Installation

Membutuhkan Python 3.9 atau lebih baru / requires Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## Menjalankan aplikasi / Run the application

```bash
streamlit run code.py
```

Buka `http://localhost:8501` di browser, lalu unggah `SUVs_Ranker_Table.csv` lewat sidebar.

## Menghasilkan tabel dan gambar hasil / Generate result tables and figures

```bash
python generate_results.py SUVs_Ranker_Table.csv
```

Hasil disimpan di folder `hasil_jurnal`. Opsi tambahan / options:

```bash
python generate_results.py SUVs_Ranker_Table.csv --k 4
python generate_results.py SUVs_Ranker_Table.csv --hp 250 --fuel 10
```

## Lisensi / License

MIT License, lihat file `LICENSE` / see the `LICENSE` file.
