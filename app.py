# app.py
from flask import Flask, request, jsonify
from utils import load_model, predict_image

app = Flask(__name__)
model = load_model()

@app.route("/")
def home():
    return "🌱 Plant Disease Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    try:
        prediction, confidence = predict_image(image, model)
        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence} %"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
