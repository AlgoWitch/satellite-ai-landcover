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

# Health check endpoint
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Backend is running"})

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
        before = request.files.get("before")
        after = request.files.get("after")
        
        if not before or not after:
            return jsonify({"error": "Missing files"}), 400

        city = request.form.get("city", "custom")

        before_path = os.path.join(UPLOADS_DIR, f"before_{city}.tif")
        after_path = os.path.join(UPLOADS_DIR, f"after_{city}.tif")

        before.save(before_path)
        after.save(after_path)
        
        print(f"Files saved: {before_path}, {after_path}")

        # Run prediction script with proper error handling
        result = subprocess.run(
            ["python", "predict.py", city],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"Prediction script failed: {error_msg}")
            return jsonify({"error": f"Prediction failed: {error_msg}"}), 500

        result_path = os.path.join(OUTPUTS_DIR, f"{city}_results.json")
        
        if not os.path.exists(result_path):
            print(f"Result file not found: {result_path}")
            return jsonify({"error": "Result file not created"}), 500

        with open(result_path) as f:
            data = json.load(f)

        return jsonify(data)

    except Exception as e:
        error_msg = str(e)
        print(f"Analysis error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=False)