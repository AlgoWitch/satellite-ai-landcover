#!/usr/bin/env python3
"""
backend/data_prep.py
Data preparation pipeline for SATELLITEX.

Reads raw feature/label GeoTIFFs and produces flat npy arrays.
Band order assumed: Band1=NIR, Band2=RED, Band3=GREEN, Band4=BLUE
(matches Sentinel-2 stack convention used in raw TIFFs).

NOTE: Raw DNs are saved as-is. Normalization (/ 10000, clip [0,1])
is applied at TRAINING time (train.py) and INFERENCE time (predict.py)
so both are always identical.
"""

import os
import numpy as np
import rasterio

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

os.makedirs(OUT, exist_ok=True)

regions = []
for f in os.listdir(RAW):
    if f.endswith("_features.tif"):
        regions.append(f.replace("_features.tif", ""))

regions.sort()
print(f"Found regions: {regions}")

all_X = []
all_y = []

for region in regions:
    feat_path = os.path.join(RAW, f"{region}_features.tif")
    lab_path  = os.path.join(RAW, f"{region}_labels.tif")

    if not os.path.exists(lab_path):
        print(f"  Skipping {region} — no label file found")
        continue

    print(f"\nProcessing: {region}")

    with rasterio.open(feat_path) as src:
        feat = src.read()                         # shape: (bands, H, W)
        feat = np.transpose(feat, (1, 2, 0))      # → (H, W, bands)
        print(f"  Features shape: {feat.shape}, dtype: {feat.dtype}")

    with rasterio.open(lab_path) as src:
        lab = src.read(1)
        print(f"  Labels   shape: {lab.shape},  dtype: {lab.dtype}")

    # Align sizes
    h = min(feat.shape[0], lab.shape[0])
    w = min(feat.shape[1], lab.shape[1])
    feat = feat[:h, :w, :]
    lab  = lab[:h, :w]

    # Flatten
    X = feat.reshape(-1, feat.shape[2]).astype(np.float32)
    y = lab.flatten()

    # --- CRITICAL FIX: fill NaN/Inf before saving ---
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove no-data pixels (label == 0 means unclassified / no-data)
    mask = y > 0
    X = X[mask]
    y = y[mask]

    print(f"  Valid pixels: {len(X):,}")
    vals, counts = np.unique(y, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"    label {v:3d}: {c:>8,d}  ({100*c/len(y):.1f}%)")

    all_X.append(X)
    all_y.append(y)

X_final = np.vstack(all_X).astype(np.float32)
y_final = np.hstack(all_y)

print(f"\nFinal dataset: X={X_final.shape}, y={y_final.shape}")
print(f"NaN in X: {np.isnan(X_final).sum()}")
print(f"Inf in X: {np.isinf(X_final).sum()}")

np.save(os.path.join(OUT, "X.npy"), X_final)
np.save(os.path.join(OUT, "y.npy"), y_final)

print("\n✅ Saved processed dataset to data/processed/")