from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

print("🚀 App is starting...")

# ✅ Model path
MODEL_PATH = "best.pt"

# ✅ DO NOT crash app if model missing
if not os.path.exists(MODEL_PATH):
    print("⚠️ WARNING: best.pt not found at startup")

# ✅ Lazy loading (IMPORTANT)
model = None


# ✅ Home route
@app.route("/")
def home():
    return "Pothole Detection API is Running 🚀"


# ✅ Detection route
@app.route("/detect", methods=["POST"])
def detect():
    global model

    try:
        # 🔥 Load model only when first request comes
        if model is None:
            print("🔄 Loading model on first request...")
            model = YOLO(MODEL_PATH, task="detect")
            print("✅ Model loaded successfully")

        # ✅ Check image
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img_bytes = file.read()

        # ✅ Convert to OpenCV image
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # ✅ Resize (VERY IMPORTANT for Render)
        img = cv2.resize(img, (640, 640))

        # ✅ Run model
        results = model(img, imgsz=320)

        detections = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    h, w = img.shape[:2]

                    detections.append({
                        "bbox": [
                            x1 / w,
                            y1 / h,
                            x2 / w,
                            y2 / h
                        ],
                        "confidence": conf
                    })

        return jsonify(detections)

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ Local run (Render ignores this, Gunicorn handles production)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)