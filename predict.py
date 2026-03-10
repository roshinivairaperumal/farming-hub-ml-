
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np, json, io
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

MODEL_KERAS = "disease_model.keras"
MODEL_H5 = "disease_model.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = (256, 256)


try:
    model = tf.keras.models.load_model(MODEL_KERAS)
    print("✅ Loaded .keras model")
except Exception:
    model = tf.keras.models.load_model(MODEL_H5)
    print("✅ Loaded .h5 model")

with open(LABELS_PATH) as f:
    class_indices = json.load(f)["class_indices"]
idx2name = {v: k for k, v in class_indices.items()}

print("🌿 ML Model loaded and ready!")

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    try:
      
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize(IMG_SIZE)
        x = np.array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)

       
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = idx2name[pred_idx]
        confidence = float(probs[pred_idx])
        
        
        result = {
            "prediction": pred_label,
            "confidence": confidence,
            "all_probabilities": {idx2name[i]: float(round(p, 4)) for i, p in enumerate(probs)},
            "status": "success"
        }
        
        print(f"✅ Prediction: {pred_label} ({confidence:.2%})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "message": "ML Model is ready for predictions"
    })

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <body>
            <h2>🌿 AgriSense ML API</h2>
            <p>ML Model is running successfully!</p>
            <p>Use <code>POST /predict</code> with an image file to get predictions.</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting ML API server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)