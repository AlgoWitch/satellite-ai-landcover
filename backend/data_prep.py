import os
import numpy as np
import rasterio

RAW = "../data/raw"
OUT = "../data/processed"

os.makedirs(OUT, exist_ok=True)

regions = []

files = os.listdir(RAW)

# detect regions automatically
for f in files:
    if f.endswith("_features.tif"):
        region = f.replace("_features.tif", "")
        regions.append(region)

all_X = []
all_y = []

for region in regions:
    feat_path = os.path.join(RAW, region + "_features.tif")
    lab_path  = os.path.join(RAW, region + "_labels.tif")

    print("Processing:", region)

    with rasterio.open(feat_path) as src:
        feat = src.read()
        feat = np.transpose(feat, (1,2,0))

    with rasterio.open(lab_path) as src:
        lab = src.read(1)

    # crop to same size
    h = min(feat.shape[0], lab.shape[0])
    w = min(feat.shape[1], lab.shape[1])

    feat = feat[:h,:w,:]
    lab  = lab[:h,:w]

    # flatten
    X = feat.reshape(-1, feat.shape[2])
    y = lab.flatten()

    # remove no-data
    mask = y > 0

    X = X[mask]
    y = y[mask]

    all_X.append(X)
    all_y.append(y)

X = np.vstack(all_X)
y = np.hstack(all_y)

print("Final dataset:", X.shape, y.shape)

np.save(os.path.join(OUT, "X.npy"), X)
np.save(os.path.join(OUT, "y.npy"), y)

print("Saved processed dataset.")