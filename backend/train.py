# =========================
# Classes:
# 0 = Vegetation
# 1 = Other Land
# 2 = Water
# =========================

import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------
# LOAD DATA
# -----------------------
X_raw = np.load("../data/processed/X.npy")
y = np.load("../data/processed/y.npy")

print("Loaded:", X_raw.shape)

# Original X columns:
# 0 = NIR
# 1 = RED
# 2 = GREEN
# 3 = BLUE

nir = X_raw[:, 0]
red = X_raw[:, 1]
green = X_raw[:, 2]
blue = X_raw[:, 3]

# Derived Features
ndvi = (nir - red) / (nir + red + 1e-5)
ndwi = (green - nir) / (green + nir + 1e-5)

# Final Feature Matrix (6 features)
X = np.stack([
    nir,
    red,
    green,
    blue,
    ndvi,
    ndwi
], axis=1)

# -----------------------
# LABEL REMAP
# WorldCover:
# 10 Trees
# 20 Shrubland
# 30 Grassland
# 40 Cropland
# 50 Built-up
# 60 Bare land
# 80 Water
# -----------------------

new_y = np.zeros_like(y)

# Vegetation
new_y[np.isin(y, [10, 20, 30, 40])] = 0

# Other Land
new_y[np.isin(y, [50, 60])] = 1

# Water
new_y[y == 80] = 2

y = new_y

mask = np.isin(y, [0, 1, 2])
X = X[mask]
y = y[mask]

# -----------------------
# SAMPLE
# -----------------------
sample_size = min(250000, len(X))

idx = np.random.choice(len(X), sample_size, replace=False)

X = X[idx]
y = y[idx]

# -----------------------
# SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# MODEL
# -----------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

print("Training...")
model.fit(X_train, y_train)

# -----------------------
# EVALUATION
# -----------------------
pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred), 5))

print(classification_report(
    y_test,
    pred,
    target_names=[
        "Vegetation",
        "Other Land",
        "Water"
    ]
))

# -----------------------
# SAVE MODEL
# -----------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/final_model.pkl")

print("Saved final model.")