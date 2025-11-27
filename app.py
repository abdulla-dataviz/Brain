# app.py  ← Save as app.py (NOT flask_api.py)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from load_member2_model import load_member2_model
from validation import validate_image
import os
import gc

app = Flask(__name__)
CORS(app)

# LOAD MODEL ONLY ONCE — THIS FIXES FREEZING
print("Loading your 92%+ accuracy model... (Please wait 10-30 seconds)")
model = load_member2_model()
print("Model loaded successfully! Now predictions are instant!")

classes = ['notumor', 'glioma', 'meningioma', 'pituitary']

@app.route('/')
def home():
    return "<h1>Brain Tumor Detection API is LIVE!</h1><p>Send POST request to /predict</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    temp_path = "temp_upload.jpg"
    file.save(temp_path)

    try:
        # Validate image
        is_valid, msg = validate_image(temp_path)
        if not is_valid:
            return jsonify({"error": msg}), 400

        # Preprocess
        img = load_img(temp_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict — super fast
        preds = model.predict(img_array, verbose=0)[0]
        confidence = float(np.max(preds) * 100)
        predicted_class = classes[np.argmax(preds)]

        probabilities = {
            "notumor": round(float(preds[0] * 100), 2),
            "glioma": round(float(preds[1] * 100), 2),
            "meningioma": round(float(preds[2] * 100), 2),
            "pituitary": round(float(preds[3] * 100), 2)
        }

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

# FIXED SYNTAX ERROR HERE
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)