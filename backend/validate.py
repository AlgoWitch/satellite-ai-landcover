#!/usr/bin/env python3
"""
backend/validate.py
Ground-truth validation for SATELLITEX predictions.

Runs the trained model on raw training TIFFs (which have known labels),
then computes accuracy, confusion matrix, and saves a side-by-side
Predicted vs Ground Truth map for visual inspection.

Usage:
    python validate.py              # validates all regions
    python validate.py gurgaon      # validates single region
"""

import os
import sys
import numpy as np
import rasterio
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_DIR   = os.path.join(ROOT_DIR, "data", "raw")
OUT_DIR   = os.path.join(ROOT_DIR, "outputs", "validation")
MODEL_PATH= os.path.join(ROOT_DIR, "models", "final_model.pkl")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants (MUST match train.py + predict.py) ───────────────────────────────
SCALE = 10000.0
CLASS_NAMES  = {0: "Vegetation", 1: "Other Land", 2: "Water"}
CLASS_COLORS = {0: [0.10,0.70,0.20], 1: [0.62,0.62,0.62], 2: [0.08,0.38,0.90]}

# ESA label → model class
LABEL_MAP = {10:0, 40:0, 20:1, 30:1, 50:1, 60:1, 80:2}

def build_features(tile: np.ndarray):
    t = tile.astype(np.float32)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    t = np.clip(t / SCALE, 0.0, 1.0)
    nir, red, green, blue = t[0], t[1], t[2], t[3]
    eps = 1e-6
    feats = [
        nir, red, green, blue,
        (nir-red)/(nir+red+eps),
        (green-nir)/(green+nir+eps),
        2.5*(nir-red)/(nir+6*red-7.5*blue+1+eps),
        (green-red)/(green+red+eps),
        (red+green+blue)/3,
        nir/(red+eps),
        red-green,
        (red-nir)/(red+nir+eps),
    ]
    return np.stack([f.ravel() for f in feats], axis=1), nir.shape

def pred_to_rgb(arr):
    rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
    for cls, col in CLASS_COLORS.items():
        rgb[arr == cls] = col
    return rgb

def validate_region(region: str, model):
    feat_path = os.path.join(RAW_DIR, f"{region}_features.tif")
    lbl_path  = os.path.join(RAW_DIR, f"{region}_labels.tif")

    if not os.path.exists(feat_path):
        print(f"  [SKIP] {region}: feature TIFF not found")
        return None
    if not os.path.exists(lbl_path):
        print(f"  [SKIP] {region}: label TIFF not found")
        return None

    print(f"\n{'='*55}")
    print(f" Validating: {region.upper()}")
    print(f"{'='*55}")

    # ── Load + predict (tile-based) ──────────────────────────────────────────
    with rasterio.open(feat_path) as src:
        H, W    = src.height, src.width
        n_bands = src.count
        print(f"  Image: {W}×{H}  |  Bands: {n_bands}")

        if n_bands < 4:
            print(f"  [SKIP] need ≥4 bands, got {n_bands}")
            return None

        from rasterio.windows import Window
        TILE = 512
        pred_map = np.zeros((H, W), dtype=np.uint8)

        for r in range(0, H, TILE):
            for c in range(0, W, TILE):
                h = min(TILE, H - r)
                w = min(TILE, W - c)
                tile = src.read(indexes=[1,2,3,4], window=Window(c,r,w,h))
                X, shp = build_features(tile)
                pred_map[r:r+h, c:c+w] = model.predict(X).reshape(shp)

    # ── Load ground truth ────────────────────────────────────────────────────
    with rasterio.open(lbl_path) as src:
        gt_raw = src.read(1)

    h = min(pred_map.shape[0], gt_raw.shape[0])
    w = min(pred_map.shape[1], gt_raw.shape[1])
    pred_map = pred_map[:h, :w]
    gt_raw   = gt_raw[:h, :w]

    # Map ESA labels → model classes
    gt_mapped = np.full(gt_raw.shape, -1, dtype=np.int8)
    for esa_cls, model_cls in LABEL_MAP.items():
        gt_mapped[gt_raw == esa_cls] = model_cls

    valid = gt_mapped >= 0
    y_true = gt_mapped[valid]
    y_pred = pred_map[valid]

    total_px  = H * W
    valid_pct = 100.0 * valid.sum() / total_px
    print(f"  Valid pixels: {valid.sum():,} / {total_px:,}  ({valid_pct:.1f}%)")

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    print(f"\n  Overall Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("\n  Confusion Matrix (rows=GT, cols=Pred):")
    print(f"  {'':15s}  {'Veg':>8s}  {'OtherLand':>9s}  {'Water':>8s}")
    for i, name in CLASS_NAMES.items():
        row = cm[i] if i < len(cm) else [0,0,0]
        print(f"  GT {name:12s}  {row[0]:>8,d}  {row[1]:>9,d}  {row[2]:>8,d}")

    print("\n  Per-class report:")
    names = [CLASS_NAMES[i] for i in range(3)]
    print(classification_report(y_true, y_pred, target_names=names, digits=3))

    # ── Predicted class distribution ─────────────────────────────────────────
    print("  Predicted distribution (full image):")
    for cls, name in CLASS_NAMES.items():
        pct = 100.0 * np.sum(pred_map == cls) / total_px
        print(f"    {name:12s}: {pct:.1f}%")

    # ── Ground truth distribution (sanity) ───────────────────────────────────
    print("\n  Ground truth distribution (valid pixels):")
    for cls, name in CLASS_NAMES.items():
        pct = 100.0 * np.sum(y_true == cls) / len(y_true)
        print(f"    {name:12s}: {pct:.1f}%")

    # ── Side-by-side visual comparison ───────────────────────────────────────
    gt_vis = np.full(gt_mapped.shape, -1, dtype=np.int8)
    gt_vis[valid] = gt_mapped[valid]

    pred_rgb = pred_to_rgb(pred_map)
    gt_rgb   = np.full((*gt_raw.shape, 3), 0.15, dtype=np.float32)  # unclassified = dark
    for cls, col in CLASS_COLORS.items():
        gt_rgb[gt_vis == cls] = col

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f1117")
    for ax, img, title in zip(axes,
                               [pred_rgb,        gt_rgb],
                               ["Model Prediction", "Ground Truth (ESA WorldCover)"]):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=13, pad=8)

    patches = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
               for k in sorted(CLASS_COLORS)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               framealpha=0.7, facecolor="#1a1d27", labelcolor="white", fontsize=11)

    acc_txt = f"Accuracy: {acc*100:.1f}%"
    fig.text(0.5, 0.96, f"{region.upper()} — {acc_txt}",
             ha="center", color="white", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    out_path = os.path.join(OUT_DIR, f"{region}_validation.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Visual saved → {out_path}")

    return {"region": region, "accuracy": acc, "n_valid": int(valid.sum())}


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"Model type: {type(model).__name__}\n")

regions_available = sorted([
    f.replace("_features.tif","")
    for f in os.listdir(RAW_DIR) if f.endswith("_features.tif")
])

if len(sys.argv) > 1:
    regions = [sys.argv[1]]
else:
    regions = regions_available

print(f"Regions to validate: {regions}")

results = []
for region in regions:
    r = validate_region(region, model)
    if r:
        results.append(r)

# ── Summary table ──────────────────────────────────────────────────────────────
if len(results) > 1:
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    print(f"{'Region':15s}  {'Accuracy':>10s}  {'Valid px':>12s}")
    print("-"*40)
    for r in results:
        print(f"{r['region']:15s}  {r['accuracy']*100:9.2f}%  {r['n_valid']:>12,d}")
    avg = np.mean([r['accuracy'] for r in results])
    print(f"\nAverage accuracy across all regions: {avg*100:.2f}%")

print("\n✅ Validation complete. Check outputs/validation/ for maps.")
