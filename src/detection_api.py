from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

def detect(frame, conf_threshold=0.5):
    """Returns detections with positional info"""
    results = model(frame, imgsz=640, verbose=False)  # Keep reliable resolution
    
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf >= conf_threshold:
                label = model.names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate center position (0=left, 1=right)
                frame_width = frame.shape[1]
                center_x = ((x1 + x2) / 2) / frame_width  
                
                detections.append({
                    "label": label,
                    "conf": conf,
                    "box": [x1, y1, x2, y2],
                    "position": center_x  # 0.0-1.0 (left-right)
                })
    return detections