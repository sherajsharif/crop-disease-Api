import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import load_model, predict_image

# ------------------- Logger Configuration -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ------------------- Flask App Setup -------------------
app = Flask(__name__)
CORS(app)

# ------------------- Model Loading -------------------
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
                logger.error("❌ Failed all attempts to load model")
    return False

# Load the model at module level so it's available for gunicorn
load_model_with_retry()

# ------------------- Routes -------------------

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "🌱 Plant Disease Detection API is running!",
        "model_status": "loaded" if model is not None else "not_loaded"
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please try again later."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    if not image.filename:
        return jsonify({"error": "No selected file"}), 400

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = os.path.splitext(image.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload a valid image file (jpg, jpeg, png, bmp)"}), 400

    try:
        prediction, confidence = predict_image(image, model)
        logger.info(f"✅ Prediction: {prediction} | Confidence: {confidence}%")
        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence} %"
        })
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Error processing image. Ensure you're uploading a valid image file."
        }), 500

# ------------------- Error Handlers -------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ------------------- Run for Local Only -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    if model is None:
        load_model_with_retry()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
