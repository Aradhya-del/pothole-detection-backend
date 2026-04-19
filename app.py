from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# ✅ Load model safely
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found: best.pt")

model = YOLO(MODEL_PATH)


# ✅ Home route
@app.route("/")
def home():
    return "Pothole Detection API is Running 🚀"


# ✅ Detection route
@app.route("/detect", methods=["POST"])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img_bytes = file.read()

        # Convert to OpenCV image
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        results = model(img)

        detections = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    height, width = img.shape[:2]

                    detections.append({
                        "bbox": [
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height
                        ],
                        "confidence": conf
                    })

        return jsonify(detections)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ IMPORTANT: Render dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)