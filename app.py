# app.py
import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import load_model, predict_image

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize model as None
model = None

def load_model_with_retry(max_retries=3):
    global model
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load model (attempt {attempt + 1}/{max_retries})")
            model = load_model()
            logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying model load...")
            else:
                logger.error("Failed all attempts to load model")
    return False

# Load model on startup
load_model_with_retry()

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "🌱 Plant Disease Detection API is running!",
        "model_status": "loaded" if model is not None else "not_loaded"
    })

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

# Health check endpoint
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    # Get port from environment variable with Render's default
    port = int(os.environ.get("PORT", 10000))
    
    # Ensure model is loaded before starting server
    if model is None:
        load_model_with_retry()
    
    # Start server
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False  # Disable reloader to prevent double model loading
    )
