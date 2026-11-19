"""
Script: predict.py
Uji coba prediksi nominal koin Rupiah secara langsung pada gambar koin nyata!
Penggunaan:
    python predict.py "path_ke_gambar_koin.jpg"
    python predict.py  (akan otomatis menguji salah satu gambar test secara acak)
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import Path

# Path direktori
NOTEBOOK_DIR = Path(r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\coin_new\notebook")
DIAGRAM_DIR  = Path(r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\coin_new\diagram")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Rp100", "Rp1000", "Rp200", "Rp500"]

# Preprocessing yang sama dengan validasi/test (anti-gepeng / CenterCrop)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_mobilenet():
    """Muat model terbaik MobileNetV2."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, len(CLASSES))
    )
    checkpoint_path = DIAGRAM_DIR / "best_mobilenetv2.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint tidak ditemukan di: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_coin(image_path, model=None):
    """Prediksi satu gambar koin dan cetak rincian probabilitasnya."""
    if model is None:
        model = load_mobilenet()

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Berkas tidak ditemukan: {image_path}")
        return None

    img = Image.open(img_path).convert("RGB")
    tensor = infer_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = probs.argmax()
    pred_label = CLASSES[pred_idx]
    pred_conf = probs[pred_idx] * 100

    print("=" * 48)
    print(f"File Target       : {img_path.name}")
    print(f"PREDIKSI NOMINAL  : {pred_label}")
    print(f"KEYAKINAN MODEL   : {pred_conf:.2f}%")
    print("-" * 48)
    print("Distribusi Probabilitas:")
    for cls_name, p in sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True):
        bar = "█" * int(p * 20)
        print(f"  {cls_name:6s} : {p*100:5.1f}%  {bar}")
    print("=" * 48)
    return pred_label, pred_conf

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default: ambil contoh acak dari test set
        csv_path = NOTEBOOK_DIR / "split.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            test_df = df[df["subset"] == "test"]
            sample_row = test_df.sample(1, random_state=None).iloc[0]
            target = sample_row["path"]
            print(f"Tidak ada input path. Mengambil contoh acak dari Test Set:")
            print(f"Label Aktual (Ground Truth): {sample_row['label']}\n")
        else:
            print("Silakan masukkan path gambar: python predict.py <path_gambar>")
            sys.exit(0)

    predict_coin(target)
