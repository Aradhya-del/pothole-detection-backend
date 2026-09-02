from flask import Flask, request, jsonify
import onnxruntime as ort
import cv2
import numpy as np
import os
import requests
import math

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.onnx"
)

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name


@app.route("/")
def home():
    return "Pothole Detection Vercel API is Running 🚀"

@app.route("/supabase-test")
def supabase_test():
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/potholes?select=id&limit=1",
            headers=SUPABASE_HEADERS,
            timeout=10
        )

        return jsonify({
            "status_code": response.status_code,
            "supabase_response": response.json()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    


def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371000

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius * c
@app.route("/nearby-potholes", methods=["GET"])
def nearby_potholes():
    try:
        latitude = request.args.get("latitude")
        longitude = request.args.get("longitude")

        print("RAW LATITUDE:", repr(latitude))
        print("RAW LONGITUDE:", repr(longitude))

        if latitude is None or longitude is None:
            return jsonify({
                "error": "latitude and longitude are required"
            }), 400

        latitude = float(latitude)
        longitude = float(longitude)

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/potholes"
            "?select=id,latitude,longitude,confidence,"
            "severity,confirmation_count,location_name,created_at",
            headers=SUPABASE_HEADERS,
            timeout=10
        )

        if response.status_code != 200:
            return jsonify({
                "error": "Failed to fetch potholes",
                "details": response.text
            }), 500

        potholes = response.json()

        nearby = []

        for pothole in potholes:

            distance = calculate_distance(
                latitude,
                longitude,
                float(pothole["latitude"]),
                float(pothole["longitude"])
            )

            # 200 meter warning radius
            if distance <= 200:

                nearby.append({
                    "id": pothole["id"],
                    "latitude": pothole["latitude"],
                    "longitude": pothole["longitude"],
                    "confidence": pothole["confidence"],
                    "severity": pothole["severity"],
                    "confirmation_count": pothole["confirmation_count"],
                    "location_name": pothole["location_name"],
                    "distance": round(distance, 2)
                })

        nearby.sort(
            key=lambda x: x["distance"]
        )

        return jsonify(nearby)

    except ValueError as e:
        return jsonify({
            "error": "Invalid latitude or longitude",
            "details": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/detect", methods=["POST"])
def detect():
    try:
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        location_name = request.form.get("location_name")

        if latitude is None or longitude is None:
            return jsonify({
                "error": "latitude and longitude are required"
            }), 400

        latitude = float(latitude)
        longitude = float(longitude)

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

            if confidence < 0.30:
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
                0.30,
                0.40
            )

            for index in indices:

                index = int(index)

                x, y, w, h = boxes[index]


                existing_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/potholes"
                    "?select=id,latitude,longitude,confirmation_count",
                    headers=SUPABASE_HEADERS,
                    timeout=10
                )

                duplicate_id = None
                duplicate_count = 0

                if existing_response.status_code == 200:

                    existing_potholes = existing_response.json()

                    for pothole in existing_potholes:

                        old_latitude = float(
                            pothole["latitude"]
                        )

                        old_longitude = float(
                            pothole["longitude"]
                        )

                        distance = calculate_distance(
                            latitude,
                            longitude,
                            old_latitude,
                            old_longitude
                        )

                        if distance <= 10:

                            duplicate_id = pothole["id"]

                            duplicate_count = (
                                pothole.get(
                                    "confirmation_count"
                                ) or 1
                            )

                            break

                if duplicate_id is not None:

                    update_response = requests.patch(
                        f"{SUPABASE_URL}/rest/v1/potholes"
                        f"?id=eq.{duplicate_id}",
                        headers=SUPABASE_HEADERS,
                        json={
                            "confirmation_count":
                                duplicate_count + 1
                        },
                        timeout=10
                    )

                    print(
                        "Duplicate pothole found:",
                        duplicate_id
                    )

                else:

                    supabase_data = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "confidence": confidences[index],
                        "severity": severities[index],
                        "confirmation_count": 1,
                        "location_name": location_name
                    }

                    insert_response = requests.post(
                        f"{SUPABASE_URL}/rest/v1/potholes",
                        headers={
                            **SUPABASE_HEADERS,
                            "Prefer": "return=minimal"
                        },
                        json=supabase_data,
                        timeout=10
                    )

                    if insert_response.status_code not in (
                        200,
                        201
                    ):
                        print(
                            "Supabase insert failed:",
                            insert_response.status_code,
                            insert_response.text
                        )

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