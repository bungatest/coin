import io
import base64
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from .model_loader import get_classifier

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")
CORS(app)  # Enable Cross-Origin Resource Sharing

# Preload model on startup
print("Memuat model MobileNetV2 di Flask...")
classifier = get_classifier()
print("Model siap digunakan!")

@app.route("/", methods=["GET"])
def index():
    """Menyajikan halaman frontend scanner"""
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    """Healthcheck endpoint untuk Postman / monitoring"""
    clf = get_classifier()
    return jsonify({
        "status": "online",
        "framework": "Flask",
        "model": clf.metadata.get("model_name", "mobilenet_v2"),
        "classes": clf.classes,
        "device": str(clf.device)
    })

@app.route("/metadata", methods=["GET"])
@app.route("/api/metadata", methods=["GET"])
def metadata():
    """Mengambil metadata kelas koin"""
    clf = get_classifier()
    return jsonify(clf.metadata)

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Endpoint utama untuk testing dari POSTMAN atau Upload File.
    Menerima file gambar via Form-Data dengan key: 'file' atau 'image'
    """
    clf = get_classifier()
    
    # 1. Cek jika dikirim via Postman multipart/form-data
    uploaded_file = None
    if "file" in request.files:
        uploaded_file = request.files["file"]
    elif "image" in request.files:
        uploaded_file = request.files["image"]
        
    if uploaded_file and uploaded_file.filename != "":
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            crop_ratio = float(request.form.get("crop_center_ratio", 0.75))
            result = clf.predict(image, crop_center_ratio=crop_ratio)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500

    # 2. Cek jika dikirim via JSON Base64
    if request.is_json:
        data = request.get_json()
        raw_b64 = data.get("image", "")
        if raw_b64:
            try:
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                image_bytes = base64.b64decode(raw_b64)
                image = Image.open(io.BytesIO(image_bytes))
                crop_ratio = float(data.get("crop_center_ratio", 1.0))
                result = clf.predict(image, crop_center_ratio=crop_ratio)
                return jsonify(result), 200
            except Exception as e:
                return jsonify({"error": f"Gagal memproses base64: {str(e)}"}), 500

    return jsonify({
        "error": "File gambar tidak ditemukan. Kirimkan file gambar dengan key 'file' di form-data Postman."
    }), 400

@app.route("/api/predict-base64", methods=["POST"])
def predict_base64():
    """Endpoint khusus streaming base64 dari canvas webcam"""
    return predict()

if __name__ == "__main__":
    port = 5000
    print("=" * 60)
    print(f"🚀 FLASK API SERVER AKTIF di http://127.0.0.1:{port}")
    print("📡 Siap di-test dari Postman di endpoint: POST /predict")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
