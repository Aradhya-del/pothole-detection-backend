from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load your trained model
model = YOLO("../runs/detect/train/weights/best.pt")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    img_bytes = file.read()

    # Convert to OpenCV image
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    return jsonify(detections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)