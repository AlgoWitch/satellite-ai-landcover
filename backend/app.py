from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess
import json

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
UPLOADS_DIR = os.path.join(BASE_DIR, '..', 'uploads')
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')

# Create directories
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# Serve frontend
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/analysis.html')
def analysis():
    return send_from_directory(FRONTEND_DIR, 'analysis.html')

# Serve output images
@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

# Serve static assets (CSS, JS, images)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        before = request.files["before"]
        after = request.files["after"]
        city = request.form.get("city", "custom")

        before_path = os.path.join(UPLOADS_DIR, f"before_{city}.tif")
        after_path = os.path.join(UPLOADS_DIR, f"after_{city}.tif")

        before.save(before_path)
        after.save(after_path)

        # Run prediction script
        result = subprocess.run(
            ["python", "predict.py", city],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print(f"Prediction error: {result.stderr}")
            return jsonify({"error": f"Prediction failed: {result.stderr}"}), 500

        result_path = os.path.join(OUTPUTS_DIR, f"{city}_results.json")

        with open(result_path) as f:
            data = json.load(f)

        return jsonify(data)

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=False)