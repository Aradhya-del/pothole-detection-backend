from flask import Flask, request, jsonify
import onnxruntime as ort
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.onnx"
)

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name


@app.route("/")
def home():
    return "Pothole Detection Vercel API is Running 🚀"


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]
        img_bytes = file.read()

        npimg = np.frombuffer(
            img_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            npimg,
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({
                "error": "Invalid image"
            }), 400

        original_height, original_width = img.shape[:2]

        img = cv2.resize(
            img,
            (640, 640)
        )

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        input_tensor = (
            img_rgb.astype(np.float32) / 255.0
        )

        input_tensor = np.transpose(
            input_tensor,
            (2, 0, 1)
        )

        input_tensor = np.expand_dims(
            input_tensor,
            axis=0
        )

        outputs = session.run(
            None,
            {
                input_name: input_tensor
            }
        )

        predictions = outputs[0][0]

        detections = []

        boxes = []
        confidences = []
        severities = []

        for prediction in predictions.T:

            x_center = float(prediction[0])
            y_center = float(prediction[1])
            box_width = float(prediction[2])
            box_height = float(prediction[3])
            confidence = float(prediction[4])

            if confidence < 0.40:
                continue

            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            x1 = max(
                0,
                min(x1, 640)
            )

            y1 = max(
                0,
                min(y1, 640)
            )

            x2 = max(
                0,
                min(x2, 640)
            )

            y2 = max(
                0,
                min(y2, 640)
            )

            box_area = (
                (x2 - x1) *
                (y2 - y1)
            )

            image_area = 640 * 640

            area_ratio = (
                box_area /
                image_area
            )

            if area_ratio >= 0.123:
                severity = "HIGH"

            elif area_ratio >= 0.05:
                severity = "MEDIUM"

            else:
                severity = "LOW"

            boxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ])

            confidences.append(
                confidence
            )

            severities.append(
                severity
            )

        if boxes:

            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                0.40,
                0.45
            )

            for index in indices:

                index = int(index)

                x, y, w, h = boxes[index]

                detections.append({
    "bbox": [
        x / 640.0,
        y / 640.0,
        (x + w) / 640.0,
        (y + h) / 640.0
    ],
    "confidence": confidences[index],
    "severity": severities[index]
})

        return jsonify(detections)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500