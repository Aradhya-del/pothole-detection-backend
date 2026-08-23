from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sqlite3
import uuid

from database import DATABASE_PATH


app = Flask(__name__)


# ============================== #
# MODEL PATH                     #
# ============================== #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)


# ============================== #
# IMAGE STORAGE                  #
# ============================== #

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "uploads",
    "potholes"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================== #
# DISTANCE CALCULATION           #
# ============================== #

def calculate_distance(lat1, lon1, lat2, lon2):

    from math import radians, sin, cos, sqrt, atan2

    R = 6371000

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


# ============================== #
# HOME ROUTE                     #
# ============================== #

@app.route("/")
def home():
    return "Pothole Detection API is Running 🚀"
# ============================== #
# GET ALL POTHOLES               #
# ============================== #
# ============================== #
# DASHBOARD ROUTE                #
# ============================== #

from flask import render_template

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ============================== #
# SERVE POTHOLE IMAGES          #
# ============================== #

@app.route("/uploads/potholes/<path:filename>")
def serve_pothole_image(filename):
    return send_from_directory(
        UPLOAD_DIR,
        filename
    )

@app.route("/potholes", methods=["GET"])
def get_potholes():

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                latitude,
                longitude,
                confidence,
                severity,
                created_at,
                confirmation_count,
                image_path
            FROM potholes
            ORDER BY id DESC
        """)

        potholes = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return jsonify(potholes)

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500

# ============================== #
# DETECTION ROUTE                #
# ============================== #

# ============================== #
# NEARBY POTHOLES API            #
# ============================== #

@app.route("/nearby-potholes", methods=["GET"])
def nearby_potholes():

    try:
        latitude = request.args.get("latitude")
        longitude = request.args.get("longitude")

        if latitude is None or longitude is None:
            return jsonify({
                "error": "latitude and longitude are required"
            }), 400

        latitude = float(latitude)
        longitude = float(longitude)

        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                latitude,
                longitude,
                confidence,
                severity,
                created_at,
                confirmation_count,
                image_path
            FROM potholes
            WHERE latitude IS NOT NULL
            AND longitude IS NOT NULL
        """)

        potholes = cursor.fetchall()
        conn.close()

        nearby = []

        for pothole in potholes:

            distance = calculate_distance(
                latitude,
                longitude,
                pothole["latitude"],
                pothole["longitude"]
            )

            # 100 meter warning radius
            if distance <= 100:

                nearby.append({
                    "id": pothole["id"],
                    "latitude": pothole["latitude"],
                    "longitude": pothole["longitude"],
                    "confidence": pothole["confidence"],
                    "severity": pothole["severity"],
                    "confirmation_count": pothole["confirmation_count"],
                    "created_at": pothole["created_at"],
                    "image_path": pothole["image_path"],
                    "distance_meters": round(distance, 2)
                })

        # Nearest pothole first
        nearby.sort(
            key=lambda pothole: pothole["distance_meters"]
        )

        return jsonify({
            "count": len(nearby),
            "potholes": nearby
        })

    except ValueError:
        return jsonify({
            "error": "Invalid latitude or longitude"
        }), 400

    except Exception as e:

        print("Nearby potholes error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/detect", methods=["POST"])
def detect():

    print("========== REQUEST RECEIVED ==========")

    try:

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        location_name = request.form.get("location_name")

        print("Latitude:", latitude)
        print("Longitude:", longitude)

        file = request.files["image"]

        # Read uploaded image
        img_bytes = file.read()

        # ============================== #
        # CONVERT IMAGE                  #
        # ============================== #

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

        # ============================== #
        # YOLO DETECTION                 #
        # ============================== #

        results = model(
            img,
            imgsz=640,
            conf=0.40,
            verbose=False
        )

        detections = []

        height, width = img.shape[:2]
        image_area = width * height

        # ============================== #
        # PROCESS DETECTIONS             #
        # ============================== #

        for r in results:

            if r.boxes is not None:

                for box in r.boxes:

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    conf = float(box.conf[0])

                    # ============================== #
                    # BOUNDING BOX AREA              #
                    # ============================== #

                    box_width = x2 - x1
                    box_height = y2 - y1

                    box_area = (
                        box_width *
                        box_height
                    )

                    area_ratio = (
                        box_area /
                        image_area
                    )

                    # ============================== #
                    # SEVERITY CLASSIFICATION        #
                    # ============================== #

                    if area_ratio >= 0.123:

                        severity = "HIGH"

                    elif area_ratio >= 0.05:

                        severity = "MEDIUM"

                    else:

                        severity = "LOW"

                    # ============================== #
                    # DATABASE                       #
                    # ============================== #

                    conn = sqlite3.connect(
                        DATABASE_PATH
                    )

                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT
                            id,
                            latitude,
                            longitude,
                            confirmation_count
                        FROM potholes
                        WHERE latitude IS NOT NULL
                        AND longitude IS NOT NULL
                    """)

                    existing_potholes = cursor.fetchall()

                    print(
                        "Existing potholes found:",
                        len(existing_potholes)
                    )

                    duplicate_id = None

                    # ============================== #
                    # DUPLICATE CHECK                #
                    # ============================== #

                    for pothole in existing_potholes:

                        pothole_id = pothole[0]
                        old_latitude = pothole[1]
                        old_longitude = pothole[2]

                        confirmation_count = (
                            pothole[3] or 1
                        )

                        distance = calculate_distance(
                            latitude,
                            longitude,
                            old_latitude,
                            old_longitude
                        )

                        print(
                            f"CHECKING ID {pothole_id}: "
                            f"distance = "
                            f"{distance:.4f} meters"
                        )

                        if distance <= 10:

                            duplicate_id = pothole_id

                            cursor.execute("""
                                UPDATE potholes
                                SET confirmation_count = ?
                                WHERE id = ?
                            """, (
                                confirmation_count + 1,
                                pothole_id
                            ))

                            print(
                                f"Duplicate pothole found: "
                                f"{distance:.2f} "
                                f"meters away"
                            )

                            break

                    # ============================== #
                    # NEW POTHOLE                    #
                    # ============================== #

                    if duplicate_id is None:

                        # Generate unique image filename
                        image_filename = (
                            f"pothole_"
                            f"{uuid.uuid4().hex}.jpg"
                        )

                        image_full_path = os.path.join(
                            UPLOAD_DIR,
                            image_filename
                        )

                        # Save original uploaded frame
                        with open(
                            image_full_path,
                            "wb"
                        ) as image_file:

                            image_file.write(
                                img_bytes
                            )

                        # Path stored in database
                        image_path = os.path.join(
                            "uploads",
                            "potholes",
                            image_filename
                        )

                        cursor.execute("""
                            INSERT INTO potholes
(
    latitude,
    longitude,
    confidence,
    severity,
    confirmation_count,
    image_path,
    location_name
)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            latitude,
                            longitude,
                            conf,
                            severity,
                            1,
                            image_path
                        ))

                        print(
                            "New pothole saved to database"
                        )

                        print(
                            "Image saved:",
                            image_path
                        )

                    # ============================== #
                    # COMMIT DATABASE               #
                    # ============================== #

                    conn.commit()
                    conn.close()

                    # ============================== #
                    # RESPONSE                       #
                    # ============================== #

                    detections.append({

                        "bbox": [
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height
                        ],

                        "confidence": conf,

                        "severity": severity
                    })

        print(
            "Detections:",
            detections
        )

        return jsonify(detections)

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================== #
# START SERVER                   #
# ============================== #

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )