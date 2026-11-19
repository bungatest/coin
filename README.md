# 🪙 Indonesian Rupiah Coin Scanner & Classification API

Sistem Klasifikasi Koin Rupiah Real-Time menggunakan Deep Learning (**MobileNetV2**) berbasis **Flask API** & **FastAPI** dengan antarmuka pemindai webcam otomatis (*auto-scanner reticle*) dan fitur aksesibilitas suara narasi (*Text-to-Speech*).

---

## 🌟 Fitur Utama
- **Real-Time Webcam Scanner**: Deteksi koin otomatis setiap 600ms dengan kotak fokus (*reticle viewfinder*), animasi centang deteksi, dan kartu hasil mengambang (*floating pill*).
- **Aksesibilitas Suara (Text-to-Speech)**: Otomatis mengucapkan nominal koin dalam Bahasa Indonesia (contoh: *"Lima Ratus Rupiah"*) untuk membantu pengguna tunanetra.
- **Dukungan Postman & Dosen Testing**: Siap diuji langsung via Postman (`POST /predict` menggunakan `multipart/form-data`). Dilengkapi file koleksi Postman siap import.
- **Dua Pilihan Backend API**:
  - **Flask API** (`run_flask.py` - Port 5000): Standar pengujian akademik kampus & Postman.
  - **FastAPI** (`run.py` - Port 8000): Asynchronous berkinerja tinggi untuk produksi.
- **MLOps Ready**: Containerized dengan Docker & Docker Compose, serta automated integration test suite (`test_api.py`).

---

## 📊 Perbandingan Model Deep Learning

Proyek ini melatih dan mengevaluasi 3 arsitektur model pada dataset koin rupiah (4 kelas: Rp 100, Rp 200, Rp 500, Rp 1.000):

| Model | Akurasi Test | Macro F1 | Ukuran Model | Latency Inferensi | Rekomendasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV2** (Dipilih) | **91.01%** | **91.38%** | **8.73 MB** | **~23.7 ms** | 🏆 **Model Terbaik Deployment** (Ringan & Cepat) |
| **GoogLeNet** | 94.38% | 94.50% | 21.5 MB | ~39.2 ms | Akurasi tertinggi, ukuran lebih besar |
| **CNN Base (Scratch)** | 77.53% | 76.98% | 1.50 MB | ~12.1 ms | Baseline pembanding |

---

## 📂 Struktur Repositori

```text
├── deploy/                               # Folder deployment aplikasi & API
│   ├── frontend/                         # Antarmuka web scanner
│   │   ├── index.html                    # Layout scanner & reticle kotak fokus
│   │   ├── style.css                     # Styling modern dark mode & glassmorphism
│   │   └── app.js                        # Controller webcam, auto-scan, & suara TTS
│   ├── backend/                          # Server API
│   │   ├── app_flask.py                  # Flask API server (Port 5000)
│   │   ├── main.py                       # FastAPI server (Port 8000)
│   │   └── model_loader.py               # Engine inferensi PyTorch MobileNetV2
│   ├── model/                            # Bobot model & metadata
│   │   ├── best_mobilenetv2.pt           # Checkpoint model terbaik (8.73 MB)
│   │   └── metadata.json                 # Label nominal, suara, dan warna
│   ├── api/                              # MLOps & Postman
│   │   ├── Rupiah_Coin_Classifier.postman_collection.json # Koleksi Postman siap import
│   │   ├── test_api.py                   # Automated test suite (5/5 PASS)
│   │   ├── Dockerfile                    # Container image
│   │   ├── docker-compose.yml            # Multi-container config
│   │   └── requirements.txt              # Dependensi server
│   ├── run_flask.py                      # Launcher Flask API (Port 5000)
│   └── run.py                            # Launcher FastAPI (Port 8000)
├── notebook/                             # Eksperimen & training Jupyter Notebooks
│   ├── 01_cnn_base.ipynb                 # Pelatihan CNN Scratch
│   ├── 02_googlenet.ipynb                # Pelatihan GoogLeNet Transfer Learning
│   ├── 03_mobilenetv2.ipynb              # Pelatihan MobileNetV2 Transfer Learning
│   ├── 04_perbandingan.ipynb             # Analisis evaluasi & grafik perbandingan
│   ├── 05_uji_prediksi.ipynb             # Notebook interaktif uji coba koin
│   ├── predict.py                        # CLI inference script
│   └── split.csv                         # Stratified dataset split (train, val, test)
├── diagram/                              # Visualisasi hasil training & evaluasi
│   ├── kurva_semua_model.png             # Perbandingan kurva loss & akurasi
│   ├── cm_semua_model.png                # Confusion matrix 3 model
│   └── ringkasan_perbandingan.csv        # Tabel ringkasan metrik
└── dataset_koin_baru/                    # Dataset gambar koin (592 sampel)
```

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Menjalankan Flask API (Rekomendasi untuk Postman & Dosen)
```bash
cd deploy
python run_flask.py
```
- Server aktif di: `http://127.0.0.1:5000`
- Browser akan terbuka otomatis menampilkan scanner koin live.

### 2. Pengujian via Postman
1. Buka aplikasi **Postman**.
2. Klik tombol **Import** ➔ pilih file:
   `deploy/api/Rupiah_Coin_Classifier.postman_collection.json`
3. Pilih request: `3. Predict Koin (Form-Data / Upload Gambar)`
4. Pada tab **Body** ➔ bagian **`file`**, pilih foto koin dari laptop Anda.
5. Klik **Send** ➔ Respons JSON klasifikasi koin akan langsung tampil.

### 3. Menjalankan FastAPI (Alternatif)
```bash
cd deploy
python run.py
```
- Server aktif di: `http://127.0.0.1:8000`
- Dokumentasi Swagger UI interaktif: `http://127.0.0.1:8000/docs`

### 4. Menjalankan MLOps Test Suite
```bash
python deploy/api/test_api.py
```
Semua 5 pengujian integrasi otomatis (healthcheck, metadata, multipart, base64, frontend route) akan dieksekusi secara otomatis.

---

## 🛠️ Instalasi Dependensi Manual
Jika dijalankan di komputer baru, install dependensi dengan:
```bash
pip install -r deploy/api/requirements.txt
```
