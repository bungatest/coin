"""
Script: 00_make_split.py
Membuat split.csv yang seimbang dan mencakup seluruh 592 gambar koin.
Stratifikasi mempertimbangkan:
  1. Label nominal (Rp100, Rp200, Rp500, Rp1000)
  2. Sisi koin (depan dan belakang / balik)
Proporsi: ~70% Train (414), ~15% Val (89), ~15% Test (89).
Seed: 42 (reproducible).
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
BASE = Path(r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\coin_new\dataset_koin_baru")
OUT_DIR = Path(r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\coin_new\notebook")

LABEL_MAP = {
    "100": "Rp100",    "balik_100": "Rp100",
    "200": "Rp200",    "balik_200": "Rp200",
    "500": "Rp500",    "balik_500": "Rp500",
    "1000": "Rp1000",  "balik_1000": "Rp1000",
}

records = []
for folder in sorted(BASE.iterdir()):
    if not folder.is_dir() or folder.name not in LABEL_MAP:
        continue
    # Muat seluruh varian file (.png, .jpg, .jpeg)
    files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    for f in sorted(files):
        records.append({
            "path": str(f),
            "label": LABEL_MAP[folder.name],
            "subfolder": folder.name,
            "filename": f.name,
            "stratum": f"{LABEL_MAP[folder.name]}_{'back' if 'balik' in folder.name else 'front'}"
        })

df = pd.DataFrame(records)
print(f"Total gambar ditemukan: {len(df)}")

# 1) Split Test (15% dari total)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(sss1.split(df, df["stratum"]))

df_trainval = df.iloc[trainval_idx].copy()
df_test = df.iloc[test_idx].copy()
df_test["subset"] = "test"

# 2) Split Train & Val dari Trainval (17.65% dari trainval = 15% dari total)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED)
train_idx, val_idx = next(sss2.split(df_trainval, df_trainval["stratum"]))

df_train = df_trainval.iloc[train_idx].copy()
df_train["subset"] = "train"
df_val = df_trainval.iloc[val_idx].copy()
df_val["subset"] = "val"

df_split = pd.concat([df_train, df_val, df_test], ignore_index=True)

print("\n=== RINGKASAN DATASET SPLIT ===")
for s in ["train", "val", "test"]:
    sub = df_split[df_split["subset"] == s]
    print(f"\n{s.upper()} ({len(sub)} gambar):")
    print(sub.groupby(["label", "subfolder"])["path"].count().to_string())

print("\nDistribusi kelas per subset:")
print(pd.crosstab(df_split["label"], df_split["subset"]))

out_path = OUT_DIR / "split.csv"
df_split[["path", "label", "subset"]].to_csv(out_path, index=False)
print(f"\nSplit berhasil disimpan ke: {out_path}")
