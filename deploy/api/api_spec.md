# 📖 Rupiah Coin Scanner - API & MLOps Documentation

Spesifikasi antarmuka API dan panduan MLOps untuk deployment model klasifikasi koin rupiah (MobileNetV2).

---

## 1. Arsitektur Folder Deploy

```text
deploy/
├── frontend/             # Antarmuka Pengguna (Webcam Scanner UI)
│   ├── index.html        # Layout scanner dengan reticle kotak fokus & bottom pill
│   ├── style.css         # Styling modern dark mode & glassmorphism
│   └── app.js            # Controller kamera, auto-scanning (600ms), TTS suara
├── backend/              # Server API FastAPI
│   ├── __init__.py
│   ├── main.py           # Endpoint REST API & serving static files
│   └── model_loader.py   # Loader PyTorch MobileNetV2 & pipeline inferensi
├── model/                # Model Checkpoint & Metadata
│   ├── best_mobilenetv2.pt  # Checkpoint MobileNetV2 terbaik (8.73 MB, Akurasi 91.01%)
│   └── metadata.json        # Mapping kelas, label teks suara, dan warna
├── api/                  # MLOps & Infrastruktur
│   ├── Dockerfile        # Container image build berbasis python:3.11-slim
│   ├── docker-compose.yml# Multi-container orchestration
│   ├── requirements.txt  # Dependencies produksi
│   ├── test_api.py       # Automated integration test suite
│   └── api_spec.md       # Dokumentasi API ini
└── run.py                # Script 1-klik untuk menjalankan server dan membuka browser
```

---

## 2. Endpoint REST API

### `GET /api/health`
Mengecek status kesehatan server dan informasi model yang sedang aktif.

- **Response:**
```json
{
  "status": "online",
  "model": "mobilenet_v2",
  "classes": ["Rp100", "Rp1000", "Rp200", "Rp500"],
  "device": "cpu"
}
```

---

### `GET /api/metadata`
Mengambil detail metadata setiap kelas koin (nominal angka, teks suara TTS, warna UI, dan deskripsi).

---

### `POST /api/predict`
Melakukan klasifikasi koin melalui upload file gambar (Multipart Form-Data).

- **Form Fields:**
  - `file`: Gambar koin (`image/jpeg`, `image/png`)
  - `crop_center_ratio`: (Opsional, default `0.75`) Memotong area tengah viewfinder.
- **Response:**
```json
{
  "prediction": "Rp500",
  "nominal": 500,
  "display": "Rp 500",
  "spoken": "Lima Ratus Rupiah",
  "color": "#8B5CF6",
  "badge_color": "#5B21B6",
  "description": "Koin Aluminium Rp 500 - Bunga Melati / Garuda Pancasila",
  "confidence": 0.9824,
  "is_confident": true,
  "probabilities": {
    "Rp100": 0.0031,
    "Rp200": 0.0042,
    "Rp500": 0.9824,
    "Rp1000": 0.0103
  },
  "inference_time_ms": 19.45
}
```

---

### `POST /api/predict-base64`
Melakukan klasifikasi koin secara real-time dari stream canvas webcam (Base64 string).

- **Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "crop_center_ratio": 1.0
}
```

---

## 3. Cara Menjalankan Aplikasi

### Opsi A: Langsung dengan Python (Paling Praktis)
Masuk ke folder `deploy` dan jalankan:
```bash
python run.py
```
Aplikasi akan otomatis menyalakan server di `http://127.0.0.1:8000` dan membuka browser Anda.

### Opsi B: Menggunakan Docker
```bash
docker compose -f api/docker-compose.yml up --build -d
```
Akses `http://localhost:8000`.

### Opsi C: Menjalankan MLOps Test Suite
```bash
python api/test_api.py
```
Menjalankan 5 pengujian otomatis untuk memvalidasi performa inferensi, format JSON, dan responsivitas API.
