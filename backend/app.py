from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess
import json
import traceback

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

# Error handler for all exceptions
@app.errorhandler(Exception)
def handle_error(error):
    """Catch all errors and return JSON instead of HTML"""
    print(f"ERROR: {str(error)}")
    traceback.print_exc()
    return jsonify({
        "error": str(error),
        "type": type(error).__name__
    }), 500

# Health check endpoint
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Backend is running"})

# Test endpoint - check if predict.py can be imported
@app.route("/api/test")
def test():
    try:
        # Try to import predict to check for import errors
        import sys
        sys.path.insert(0, BASE_DIR)
        
        test_data = {
            "status": "ok",
            "base_dir": BASE_DIR,
            "uploads_dir": UPLOADS_DIR,
            "outputs_dir": OUTPUTS_DIR,
            "uploads_exists": os.path.exists(UPLOADS_DIR),
            "outputs_exists": os.path.exists(OUTPUTS_DIR),
        }
        
        # Check if rasterio can be imported
        try:
            import rasterio
            test_data["rasterio"] = "✓ installed"
        except ImportError as e:
            test_data["rasterio"] = f"✗ {str(e)}"
        
        # Check if numpy can be imported
        try:
            import numpy
            test_data["numpy"] = "✓ installed"
        except ImportError as e:
            test_data["numpy"] = f"✗ {str(e)}"
        
        return jsonify(test_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        print("=== ANALYZE REQUEST START ===")
        
        before = request.files.get("before")
        after = request.files.get("after")
        
        if not before or not after:
            print("ERROR: Missing files")
            return jsonify({"error": "Missing files"}), 400

        city = request.form.get("city", "custom")
        print(f"City: {city}")

        before_path = os.path.join(UPLOADS_DIR, f"before_{city}.tif")
        after_path = os.path.join(UPLOADS_DIR, f"after_{city}.tif")

        print(f"Saving files to: {before_path}, {after_path}")
        before.save(before_path)
        after.save(after_path)
        
        if not os.path.exists(before_path):
            print(f"ERROR: Before file not saved: {before_path}")
            return jsonify({"error": "Before file not saved"}), 500
        
        if not os.path.exists(after_path):
            print(f"ERROR: After file not saved: {after_path}")
            return jsonify({"error": "After file not saved"}), 500
        
        print(f"Files exist: before={os.path.getsize(before_path)} bytes, after={os.path.getsize(after_path)} bytes")

        # Run prediction script
        print(f"Running: python predict.py {city} from {BASE_DIR}")
        result = subprocess.run(
            ["python", "predict.py", city],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )

        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"ERROR: Prediction failed")
            return jsonify({"error": f"Prediction failed: {error_msg}"}), 500

        result_path = os.path.join(OUTPUTS_DIR, f"{city}_results.json")
        
        if not os.path.exists(result_path):
            print(f"ERROR: Result file not found: {result_path}")
            return jsonify({"error": "Result file not created"}), 500

        print(f"Loading results from: {result_path}")
        with open(result_path) as f:
            data = json.load(f)

        print("=== ANALYZE REQUEST SUCCESS ===")
        return jsonify(data)

    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg, "type": type(e).__name__}), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=False)