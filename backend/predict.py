#!/usr/bin/env python3
"""
backend/predict.py
Production Inference Pipeline — SATELLITEX

Callable as a module from app.py:
    from predict import run_prediction
    results = run_prediction(before_path, after_path, city, output_dir, model)

Or as a standalone CLI script:
    python predict.py <city>

Normalization MUST match train.py exactly:
    tile / 10000.0, clipped to [0, 1]
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, json, sys, gc, joblib

try:
    from scipy.ndimage import binary_closing, binary_opening
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[predict] WARNING: scipy not installed — change maps will be noisier")

# ── Preprocessing Constants (MUST match train.py) ──────────────────────────────
SCALE_FACTOR = 10000.0
CLIP_MIN     = 0.0
CLIP_MAX     = 1.0
TILE_SIZE    = 512

# ── Class Palette ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: [0.10, 0.70, 0.20],   # Vegetation — Green
    1: [0.62, 0.62, 0.62],   # Other Land — Gray
    2: [0.08, 0.38, 0.90],   # Water      — Blue
}
CLASS_NAMES = {0: "Vegetation", 1: "Other Land", 2: "Water"}


# ── Feature Engineering ────────────────────────────────────────────────────────
def build_features(tile: np.ndarray):
    """
    tile : ndarray shape (>=4, H, W) — raw DN values (uint16 or float)
    Returns (X shape [H*W, 12], tile_shape (H, W))

    Normalization applied here exactly mirrors train.py.
    """
    t = tile.astype(np.float32)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    t = t / SCALE_FACTOR
    t = np.clip(t, CLIP_MIN, CLIP_MAX)

    nir   = t[0];  red   = t[1]
    green = t[2];  blue  = t[3]
    eps   = 1e-6

    ndvi       = (nir - red)   / (nir + red   + eps)
    ndwi       = (green - nir) / (green + nir + eps)
    evi        = 2.5*(nir-red) / (nir + 6*red - 7.5*blue + 1 + eps)
    mndwi      = (green - red) / (green + red + eps)
    brightness = (red + green + blue) / 3.0
    nir_r_rat  = nir / (red + eps)
    red_green  = red - green
    soil_idx   = (red - nir) / (red + nir + eps)

    H, W = nir.shape
    X = np.stack([
        nir.ravel(), red.ravel(), green.ravel(), blue.ravel(),
        ndvi.ravel(), ndwi.ravel(), evi.ravel(), mndwi.ravel(),
        brightness.ravel(), nir_r_rat.ravel(), red_green.ravel(), soil_idx.ravel()
    ], axis=1)
    return X, (H, W)


# ── Tile-Based Inference ───────────────────────────────────────────────────────
def predict_image(path: str, model, tile_size: int = TILE_SIZE) -> np.ndarray:
    """
    Memory-safe tile-based inference on a GeoTIFF of any size.
    Reads the first 4 bands (NIR, RED, GREEN, BLUE).
    """
    with rasterio.open(path) as src:
        n_bands = src.count
        H, W    = src.height, src.width

        if n_bands < 4:
            raise ValueError(
                f"TIFF '{os.path.basename(path)}' has {n_bands} band(s); "
                "need at least 4 (NIR, RED, GREEN, BLUE)."
            )

        out = np.zeros((H, W), dtype=np.uint8)
        band_idx = [1, 2, 3, 4]   # rasterio is 1-indexed

        for r in range(0, H, tile_size):
            for c in range(0, W, tile_size):
                h = min(tile_size, H - r)
                w = min(tile_size, W - c)

                tile = src.read(indexes=band_idx, window=Window(c, r, w, h))
                X, shp = build_features(tile)
                pred   = model.predict(X).reshape(shp)
                out[r:r+h, c:c+w] = pred

                del tile, X, pred
                gc.collect()

    return out


# ── Change Detection ───────────────────────────────────────────────────────────
def compute_change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Returns boolean change mask.
    Morphological opening (removes isolated noise pixels) then
    closing (fills small gaps) for a cleaner map.
    """
    raw = (before != after)
    if not HAS_SCIPY or raw.sum() == 0:
        return raw.astype(bool)
    struct  = np.ones((3, 3), dtype=bool)
    cleaned = binary_opening(raw, structure=struct, iterations=1)
    cleaned = binary_closing(cleaned, structure=struct, iterations=1)
    return cleaned.astype(bool)


# ── Visualization ──────────────────────────────────────────────────────────────
def _make_legend(ax, items, facecolor="#1a1d27"):
    patches = [mpatches.Patch(color=c, label=l) for l, c in items]
    ax.legend(handles=patches, loc="lower right",
              framealpha=0.75, fontsize=9,
              facecolor=facecolor, labelcolor="white")


def save_land_map(pred: np.ndarray, path: str, title: str = "Land Cover"):
    H, W = pred.shape
    rgb  = np.zeros((H, W, 3), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        rgb[pred == cls_id] = color

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0f1117")
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    _make_legend(ax, [(CLASS_NAMES[k], CLASS_COLORS[k]) for k in sorted(CLASS_COLORS)])
    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_change_map(before: np.ndarray, after: np.ndarray, path: str):
    change = compute_change(before, after)
    H, W   = change.shape

    # Background: light grey (no change)
    rgb = np.full((H, W, 3), 0.92, dtype=np.float32)

    # Changed pixels — tinted by what was LOST
    for cls_id, color in CLASS_COLORS.items():
        lost = change & (before == cls_id) & (after != cls_id)
        if lost.any():
            c = np.array(color)
            rgb[lost] = np.clip(c * 0.55 + np.array([0.45, 0.05, 0.05]), 0, 1)

    # Fallback: pure red for any remaining changed pixels not tinted
    untinted = change & np.all(rgb == 0.92, axis=-1)
    rgb[untinted] = [0.90, 0.15, 0.15]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0f1117")
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("Change Detection Map", color="white", fontsize=13, pad=10)
    pct = 100.0 * change.sum() / change.size
    ax.text(0.02, 0.02, f"Changed: {pct:.1f}% of area",
            transform=ax.transAxes, color="white", fontsize=9,
            bbox=dict(facecolor="#0f1117", alpha=0.7, edgecolor="none"))
    _make_legend(ax, [("No Change", [0.92, 0.92, 0.92]), ("Changed", [0.90, 0.15, 0.15])])
    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_stats(pred: np.ndarray) -> dict:
    total = pred.size
    return {
        "Vegetation": round(float(np.sum(pred == 0) / total * 100), 2),
        "Other Land": round(float(np.sum(pred == 1) / total * 100), 2),
        "Water":      round(float(np.sum(pred == 2) / total * 100), 2),
    }


# ── Main callable (used by app.py) ────────────────────────────────────────────
def run_prediction(before_path: str, after_path: str, city: str,
                   output_dir: str, model) -> dict:
    """
    Full pipeline: inference → maps → stats → JSON.
    Returns the results dict (same shape the frontend expects).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[predict] city={city}")
    print(f"[predict] before={before_path}  ({os.path.getsize(before_path)/1e6:.1f} MB)")
    print(f"[predict] after ={after_path}   ({os.path.getsize(after_path)/1e6:.1f} MB)")

    before_pred = predict_image(before_path, model)
    after_pred  = predict_image(after_path,  model)

    before_map = os.path.join(output_dir, f"{city}_before_map.png")
    after_map  = os.path.join(output_dir, f"{city}_after_map.png")
    change_map = os.path.join(output_dir, f"{city}_change_map.png")

    save_land_map(before_pred, before_map, title="Before — Land Cover")
    save_land_map(after_pred,  after_map,  title="After  — Land Cover")
    save_change_map(before_pred, after_pred, change_map)

    b = compute_stats(before_pred)
    a = compute_stats(after_pred)

    results = {
        "city":   city,
        "before": b,
        "after":  a,
        "change": {
            "Vegetation Change %": round(a["Vegetation"] - b["Vegetation"], 2),
            "Other Land Change %": round(a["Other Land"] - b["Other Land"], 2),
            "Water Change %":      round(a["Water"]      - b["Water"],      2),
        },
        "maps": {
            "before": f"/outputs/{city}_before_map.png",
            "after":  f"/outputs/{city}_after_map.png",
            "change": f"/outputs/{city}_change_map.png",
        }
    }

    result_json = os.path.join(output_dir, f"{city}_results.json")
    with open(result_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[predict] Done → {result_json}")
    return results


# ── Standalone CLI mode ────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR   = os.path.abspath(os.path.join(BASE_DIR, ".."))
    UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "final_model.pkl")

    city = sys.argv[1] if len(sys.argv) > 1 else "custom"
    before_path = os.path.join(UPLOAD_DIR, f"before_{city}.tif")
    after_path  = os.path.join(UPLOAD_DIR, f"after_{city}.tif")

    print(f"Loading model from {MODEL_PATH} ...")
    mdl = joblib.load(MODEL_PATH)

    run_prediction(before_path, after_path, city, OUTPUT_DIR, mdl)
    print("Prediction complete.")