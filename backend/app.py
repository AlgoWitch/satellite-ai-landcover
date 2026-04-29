"""
backend/app.py
Production Flask Backend — SATELLITEX

Key changes from original:
- Model loaded ONCE at startup (not per-request via subprocess)
- predict.run_prediction() called in-process (no subprocess fragility)
- Timeout raised to 300s for large TIFFs (60-150 MB)
- File size guard (200 MB max)
- Proper error messages returned as JSON always
- CORS fixed for both local and Render production
"""

import os
import sys
import json
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
UPLOADS_DIR  = os.path.join(ROOT_DIR, "uploads")
OUTPUTS_DIR  = os.path.join(ROOT_DIR, "outputs")
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "final_model.pkl")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Import predict module (in-process, no subprocess) ─────────────────────────
sys.path.insert(0, BASE_DIR)
from predict import run_prediction   # noqa: E402

# ── Load Model Once at Startup ─────────────────────────────────────────────────
print(f"[app] Loading model from {MODEL_PATH} ...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run backend/train.py first to generate final_model.pkl"
    )
MODEL = joblib.load(MODEL_PATH)
print(f"[app] Model loaded — {type(MODEL).__name__}")

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.errorhandler(Exception)
def handle_error(error):
    print(f"[app] UNHANDLED ERROR: {error}")
    traceback.print_exc()
    return jsonify({"error": str(error), "type": type(error).__name__}), 500


# ── Health Check ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": type(MODEL).__name__,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
    })


# ── System Info ────────────────────────────────────────────────────────────────
@app.route("/api/test")
def test():
    info = {
        "status": "ok",
        "base_dir": BASE_DIR,
        "uploads_dir": UPLOADS_DIR,
        "outputs_dir": OUTPUTS_DIR,
        "model_loaded": MODEL is not None,
    }
    try:
        import rasterio
        info["rasterio"] = rasterio.__version__
    except ImportError as e:
        info["rasterio"] = f"NOT INSTALLED: {e}"
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except ImportError as e:
        info["numpy"] = f"NOT INSTALLED: {e}"
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = "NOT INSTALLED (change maps will be noisier)"
    return jsonify(info)


# ── Serve Frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/analysis.html")
def analysis():
    return send_from_directory(FRONTEND_DIR, "analysis.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Serve Output Images ────────────────────────────────────────────────────────
@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


# ── Main Analyze Endpoint ──────────────────────────────────────────────────────
MAX_FILE_MB = 200

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        print("=== /analyze REQUEST START ===")

        before_file = request.files.get("before")
        after_file  = request.files.get("after")

        if not before_file or not after_file:
            return jsonify({"error": "Missing files. Upload both 'before' and 'after' TIFFs."}), 400

        city = request.form.get("city", "custom").strip() or "custom"
        # Sanitize city name to safe filename
        city = "".join(c for c in city if c.isalnum() or c in "-_").lower() or "custom"

        before_path = os.path.join(UPLOADS_DIR, f"before_{city}.tif")
        after_path  = os.path.join(UPLOADS_DIR, f"after_{city}.tif")

        # Save uploads
        before_file.save(before_path)
        after_file.save(after_path)

        # File size guard
        for label, fpath in [("before", before_path), ("after", after_path)]:
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {label}: {size_mb:.1f} MB")
            if size_mb > MAX_FILE_MB:
                return jsonify({
                    "error": f"File '{label}' is {size_mb:.0f} MB — maximum allowed is {MAX_FILE_MB} MB."
                }), 413

        # In-process prediction (model already loaded)
        results = run_prediction(
            before_path=before_path,
            after_path=after_path,
            city=city,
            output_dir=OUTPUTS_DIR,
            model=MODEL
        )

        print("=== /analyze REQUEST SUCCESS ===")
        return jsonify(results)

    except ValueError as e:
        # e.g. wrong band count
        print(f"[app] ValueError: {e}")
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        print(f"[app] ERROR in /analyze: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    print(f"[app] Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)