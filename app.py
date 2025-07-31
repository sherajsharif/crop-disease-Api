# app.py
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import load_model, predict_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route("/")
def home():
    if model is None:
        return "⚠️ API is running but model is not loaded properly!", 503
    return "🌱 Plant Disease Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Please try again later"}), 503

    # Check if image is in request
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    
    # Validate image file
    if not image.filename:
        return jsonify({"error": "No selected file"}), 400
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = os.path.splitext(image.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload a valid image file (jpg, jpeg, png, bmp)"}), 400
        
    try:
        # Process image and get prediction
        prediction, confidence = predict_image(image, model)
        
        # Log successful prediction
        logger.info(f"Successfully predicted {prediction} with confidence {confidence}%")
        
        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence} %"
        })
    except Exception as e:
        # Log the error with stack trace
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Error processing image. Please ensure you're uploading a valid image file."
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
