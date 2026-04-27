# ==========================================
# backend/predict.py
# Satellite Change Detection - Prediction
# ==========================================

import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server
import matplotlib.pyplot as plt
import os
import json
import sys

# ---------------------------------
# CONFIG - Use absolute paths
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

city = sys.argv[1] if len(sys.argv) > 1 else "custom"

before_path = os.path.join(DATA_DIR, f"before_{city}.tif")
after_path = os.path.join(DATA_DIR, f"after_{city}.tif")

# ---------------------------------
# LOAD TIFF
# ---------------------------------
def load_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img, (1,2,0))
        img = img.astype(np.float32)

    img = np.nan_to_num(img, nan=0.0)
    img = img / 10000.0
    img = np.clip(img, 0, 1)

    return img

# ---------------------------------
# CLASSIFY
# 0 Vegetation
# 1 Other Land
# 2 Water
# ---------------------------------
def classify_scene(img):

    nir = img[:, :, 0]
    red = img[:, :, 1]
    green = img[:, :, 2]

    ndvi = (nir - red) / (nir + red + 1e-5)
    ndwi = (green - nir) / (green + nir + 1e-5)

    pred = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

    pred[ndvi > 0.28] = 0
    pred[ndwi > 0.12] = 2

    return pred

# ---------------------------------
# SAVE CLASS MAP
# ---------------------------------
def save_map(pred, path):

    rgb = np.zeros((pred.shape[0], pred.shape[1], 3))

    rgb[pred == 0] = [0.0, 0.75, 0.2]   # vegetation
    rgb[pred == 1] = [0.75, 0.75, 0.75] # land
    rgb[pred == 2] = [0.0, 0.4, 1.0]    # water

    plt.figure(figsize=(8,8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

# ---------------------------------
# SAVE CHANGE MAP
# ---------------------------------
def save_change_map(before, after, path):

    rgb = np.zeros((before.shape[0], before.shape[1], 3))

    changed = before != after

    rgb[:] = [0.92, 0.92, 0.92]     # no change
    rgb[changed] = [1.0, 0.15, 0.15] # changed

    plt.figure(figsize=(8,8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

# ---------------------------------
# PERCENTAGES
# ---------------------------------
def stats(pred):

    total = pred.size

    return {
        "Vegetation": round(np.sum(pred == 0)/total*100,2),
        "Other Land": round(np.sum(pred == 1)/total*100,2),
        "Water": round(np.sum(pred == 2)/total*100,2)
    }

# ---------------------------------
# MAIN - Run prediction
# ---------------------------------
try:
    if not os.path.exists(before_path):
        print(f"Error: Before file not found: {before_path}")
        exit(1)
    
    if not os.path.exists(after_path):
        print(f"Error: After file not found: {after_path}")
        exit(1)
    
    print(f"Loading: {before_path}")
    before_img = load_tif(before_path)
    print(f"Loading: {after_path}")
    after_img = load_tif(after_path)

    before = classify_scene(before_img)
    after = classify_scene(after_img)

    save_map(before, os.path.join(OUTPUT_DIR, f"{city}_before_map.png"))
    save_map(after, os.path.join(OUTPUT_DIR, f"{city}_after_map.png"))
    save_change_map(before, after, os.path.join(OUTPUT_DIR, f"{city}_change_map.png"))

    b = stats(before)
    a = stats(after)

    change = {
        "Vegetation Change %": round(a["Vegetation"]-b["Vegetation"],2),
        "Other Land Change %": round(a["Other Land"]-b["Other Land"],2),
        "Water Change %": round(a["Water"]-b["Water"],2)
    }

    results = {
        "city": city,
        "before": b,
        "after": a,
        "change": change,
        "maps": {
            "before": f"/outputs/{city}_before_map.png",
            "after": f"/outputs/{city}_after_map.png",
            "change": f"/outputs/{city}_change_map.png"
        }
    }

    with open(os.path.join(OUTPUT_DIR, f"{city}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("✓ Prediction complete")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)