#!/usr/bin/env python3
"""
backend/train.py
Production Training Pipeline — SATELLITEX

KEY FIX: Applies /10000 normalization + clip[0,1] BEFORE feature
engineering, exactly matching predict.py. This was the primary cause
of all wrong predictions in the original code.

Uses ExtraTreesClassifier: faster training, smaller model file,
comparable accuracy to RandomForest on spectral satellite data.
"""

import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Preprocessing Constants ────────────────────────────────────────────────────
# CRITICAL: These MUST be identical to predict.py
SCALE_FACTOR = 10000.0
CLIP_MIN     = 0.0
CLIP_MAX     = 1.0

# ── Load Raw Features ──────────────────────────────────────────────────────────
print("Loading dataset...")
X_raw = np.load(os.path.join(DATA_DIR, "X.npy"))
y_raw = np.load(os.path.join(DATA_DIR, "y.npy"))

print(f"Loaded: X={X_raw.shape}  y={y_raw.shape}")
print(f"X dtype: {X_raw.dtype}   y dtype: {y_raw.dtype}")

# ── CRITICAL FIX: Normalize to match inference pipeline ───────────────────────
# predict.py does /10000 + clip before computing indices.
# Training MUST do the same or every decision boundary is wrong.
print("\nNormalizing features (÷10000, clip [0,1])...")
X_norm = X_raw.astype(np.float32)
X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1.0, neginf=0.0)
X_norm = X_norm / SCALE_FACTOR
X_norm = np.clip(X_norm, CLIP_MIN, CLIP_MAX)

# ── Band Assignment (4 bands: NIR, RED, GREEN, BLUE) ──────────────────────────
nir   = X_norm[:, 0]   # Band 1 — NIR (B8 in Sentinel-2)
red   = X_norm[:, 1]   # Band 2 — Red (B4)
green = X_norm[:, 2]   # Band 3 — Green (B3)
blue  = X_norm[:, 3]   # Band 4 — Blue (B2)

# ── Feature Engineering (12 features) ─────────────────────────────────────────
eps = 1e-6
ndvi       = (nir - red)   / (nir + red   + eps)          # Vegetation
ndwi       = (green - nir) / (green + nir + eps)           # Water
evi        = 2.5*(nir-red) / (nir + 6*red - 7.5*blue + 1 + eps)  # Enhanced Veg
mndwi      = (green - red) / (green + red + eps)            # Modified Water
brightness = (red + green + blue) / 3.0
nir_r_rat  = nir / (red + eps)
red_green  = red - green
soil_idx   = (red - nir) / (red + nir + eps)               # Bare Soil Index

X = np.stack([
    nir, red, green, blue,
    ndvi, ndwi, evi, mndwi,
    brightness, nir_r_rat, red_green, soil_idx
], axis=1).astype(np.float32)

print(f"Feature matrix: {X.shape}  (12 features per pixel)")
del X_raw, X_norm, nir, red, green, blue  # free memory

# ── Label Mapping ──────────────────────────────────────────────────────────────
# ESA WorldCover classes in training data:
#  10 = Trees            → 0 Vegetation
#  20 = Shrubland        → 1 Other Land
#  30 = Grassland        → 1 Other Land
#  40 = Cropland         → 0 Vegetation  (vegetated, spectral overlap with trees)
#  50 = Built-up         → 1 Other Land
#  60 = Bare/Sparse veg  → 1 Other Land
#  80 = Permanent Water  → 2 Water
#  90 = Snow/Ice         → DROPPED (~0.08%, Corbett only)
CLASS_NAMES = ["Vegetation", "Other Land", "Water"]

y = np.full(y_raw.shape, -1, dtype=np.int8)
y[np.isin(y_raw, [10, 40])]         = 0   # Vegetation
y[np.isin(y_raw, [20, 30, 50, 60])] = 1   # Other Land
y[y_raw == 80]                       = 2   # Water

valid = y >= 0
X = X[valid]
y = y[valid]
del y_raw

print("\nOriginal class distribution:")
for i, name in enumerate(CLASS_NAMES):
    n = int(np.sum(y == i))
    print(f"  {name:12s}: {n:>10,d}  ({100*n/len(y):.1f}%)")

# ── Balanced Sampling ──────────────────────────────────────────────────────────
PER_CLASS = 200_000
rng = np.random.default_rng(42)
idx_list = []
for cls in range(3):
    cls_idx = np.where(y == cls)[0]
    n = min(PER_CLASS, len(cls_idx))
    idx_list.append(rng.choice(cls_idx, size=n, replace=False))

idx = np.concatenate(idx_list)
rng.shuffle(idx)
X, y = X[idx], y[idx]

print("\nBalanced training set:")
for i, name in enumerate(CLASS_NAMES):
    n = int(np.sum(y == i))
    print(f"  {name:12s}: {n:>10,d}")

# ── Train / Test Split ─────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_tr):,}   Test: {len(X_te):,}")

# ── Model ──────────────────────────────────────────────────────────────────────
# ExtraTrees: random splits → faster than RF, smaller .pkl, same accuracy
# n_estimators=150 + max_depth=20 → ~100-200 MB model (safe for Render)
model = ExtraTreesClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample"
)

print("\nTraining ExtraTreesClassifier...")
print("(~5-10 min on CPU)")
model.fit(X_tr, y_tr)

# ── Evaluation ─────────────────────────────────────────────────────────────────
pred = model.predict(X_te)
acc  = accuracy_score(y_te, pred)
print(f"\n=== EVALUATION ===")
print(f"Overall Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
print("\nConfusion Matrix:")
print(confusion_matrix(y_te, pred))
print("\nClassification Report:")
print(classification_report(y_te, pred, target_names=CLASS_NAMES))

# ── Save Model ─────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "final_model.pkl")
joblib.dump(model, model_path, compress=3)
size_mb = os.path.getsize(model_path) / 1e6
print(f"\nModel saved: {model_path}  ({size_mb:.1f} MB)")

# ── Save Preprocessing Config ──────────────────────────────────────────────────
config = {
    "scale_factor":  SCALE_FACTOR,
    "clip_min":      CLIP_MIN,
    "clip_max":      CLIP_MAX,
    "band_order":    ["NIR", "RED", "GREEN", "BLUE"],
    "n_features":    12,
    "feature_names": [
        "nir", "red", "green", "blue",
        "ndvi", "ndwi", "evi", "mndwi",
        "brightness", "nir_red_ratio", "red_green", "soil_index"
    ],
    "classes":       {0: "Vegetation", 1: "Other Land", 2: "Water"},
    "n_estimators":  150,
    "max_depth":     20,
    "accuracy":      float(round(acc, 4))
}
cfg_path = os.path.join(MODEL_DIR, "preprocessing_config.json")
with open(cfg_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved: {cfg_path}")
print("\n✅ Training complete.")