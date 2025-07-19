import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os # Import the os module for path manipulation

app = Flask(__name__)
CORS(app)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras') # Construct the full path to the model

# Load the Keras model when the application starts
try:
    # Use the constructed MODEL_PATH to ensure correct loading
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Keras model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None # Set model to None to handle cases where it fails to load

@app.route('/predict_tumor', methods=['POST'])
def predict_tumor():
    """
    API endpoint to receive an image, process it, and predict if a tumor is present.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please check backend logs."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in the request."}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file:
        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            tumor_probability = predictions[0][0]
            result = "Tumor Detected" if tumor_probability > 0.5 else "No Tumor Detected"

            return jsonify({
                "prediction": result,
                "probability": float(tumor_probability)
            })
        except Exception as e:
            return jsonify({"error": f"Error processing image or making prediction: {e}"}), 500
    return jsonify({"error": "An unexpected error occurred during file processing."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
