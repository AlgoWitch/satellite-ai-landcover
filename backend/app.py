from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "../uploads"
OUTPUT_FOLDER = "../outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze():

    before = request.files["before"]
    after = request.files["after"]

    city = request.form.get("city", "custom")

    before_path = f"{UPLOAD_FOLDER}/before_{city}.tif"
    after_path = f"{UPLOAD_FOLDER}/after_{city}.tif"

    before.save(before_path)
    after.save(after_path)

    # Run prediction script
    subprocess.run(["python", "predict.py", city])

    result_path = f"{OUTPUT_FOLDER}/{city}_results.json"

    with open(result_path) as f:
        data = json.load(f)

    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)