# Plant Disease Detection API

This is a Flask-based API for detecting diseases in plants using deep learning. The API can identify diseases in Corn, Potato, Rice, Tomato, and Wheat plants.

## Features

- Disease detection for multiple crop types
- Fast and accurate predictions using ResNet18 model
- Easy-to-use REST API interface

## Setup

1. Clone the repository:
```bash
git clone [your-repository-url]
cd Model_API
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
python app.py
```

## API Usage

### Endpoint: `/predict`
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `image` (file)

Example Response:
```json
{
    "prediction": "Tomato___Healthy",
    "confidence": "98.45 %"
}
```

## Supported Plant Diseases

The model can detect 25 different classes including:
- Corn diseases (Common Rust, Gray Leaf Spot, Northern Leaf Blight)
- Potato diseases (Early Blight, Late Blight)
- Rice diseases (Brown Spot, Leaf Blast, Neck Blast)
- Tomato diseases (Bacterial spot, Early blight, Late blight, Leaf Mold, etc.)
- Wheat diseases (Brown Rust, Yellow Rust)
- Healthy plants for each crop type
