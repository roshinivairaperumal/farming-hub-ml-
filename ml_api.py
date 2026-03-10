
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np, json, io, requests
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)


MODEL_PATH = "disease_model.h5"
LABELS_PATH = "labels.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    class_indices = json.load(f)["class_indices"]
idx2name = {v: k for k, v in class_indices.items()}

ESP32_CAM_URL = "http://10.37.1.217"

def prepare_image(img, target_size=(256, 256)):
    img = img.convert("RGB").resize(target_size)
    x = np.asarray(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)
    return x

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        x = prepare_image(img)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        
        return jsonify({
            "prediction": idx2name[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {idx2name[i]: float(round(p, 4)) for i, p in enumerate(probs)},
            "all_predictions": [
                {
                    "disease": idx2name[i],
                    "confidence": float(round(p, 4)),
                    "percentage": float(round(p * 100, 2))
                }
                for i, p in enumerate(probs)
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/capture-and-analyze', methods=['GET'])
def capture_and_analyze():
    """Capture from ESP32-CAM and analyze directly"""
    try:
        
        response = requests.get(f"{ESP32_CAM_URL}/capture", timeout=10)
        if response.status_code != 200:
            return jsonify({"error": "Failed to capture from ESP32-CAM"}), 500
        
        img = Image.open(io.BytesIO(response.content))
        x = prepare_image(img)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return jsonify({
            "prediction": idx2name[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {idx2name[i]: float(round(p, 4)) for i, p in enumerate(probs)},
            "image_data": img_byte_arr.getvalue().hex()  # Send image as hex for demo
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "esp32_connected": check_esp32_connection()
    })

def check_esp32_connection():
    try:
        response = requests.get(f"{ESP32_CAM_URL}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)