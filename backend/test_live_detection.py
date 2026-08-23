import sys
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(os.path.join(os.path.dirname(__file__), '..', 'best.pt'))

if len(sys.argv) < 2:
    print(json.dumps([]))
    sys.exit(0)

image_path = sys.argv[1]
img = cv2.imread(image_path)
if img is None:
    print(json.dumps([]))
    sys.exit(0)

results = model(img, stream=False)
detections = []
for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            h, w = img.shape[:2]
            if conf >= 0.1:
                detections.append({
                    'bbox': [x1 / w, y1 / h, x2 / w, y2 / h],
                    'confidence': conf,
                })

print(json.dumps(detections))
