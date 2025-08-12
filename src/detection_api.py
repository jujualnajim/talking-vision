from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect(frame, conf_threshold=0.5):
    results = model(frame, verbose=False)[0]
    detections = []
    
    for box in results.boxes:
        conf = float(box.conf)
        if conf >= conf_threshold:
            detections.append({
                "label": model.names[int(box.cls)],
                "conf": conf,
                "box": [int(x) for x in box.xyxy[0].tolist()]
            })
    return detections