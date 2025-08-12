import cv2
import time
import pytesseract
import numpy as np
from text_to_speech import SpeechEngine
from detection_api import detect  # Import the detect function

# Configuration
CONF_THRESHOLD = 0.5
SPEAK_INTERVAL = 3.0
HAZARD_OBJECTS = ["stairs", "knife", "car", "fire"]
POSITION_THRESHOLDS = {
    "left": (0.0, 0.4),
    "center": (0.4, 0.6),
    "right": (0.6, 1.0)
}

class OCRProcessor:
    def __init__(self):
        # Auto-detect Tesseract path
        self.tesseract_cmd = self.find_tesseract()
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        self.tess_config = r'--oem 3 --psm 6 -l eng+fra+spa'  # English + French + Spanish

    def find_tesseract(self):
        """Search for Tesseract in common locations"""
        import os
        possible_paths = [
            '/usr/local/bin/tesseract',  # Intel Mac Homebrew
            '/opt/homebrew/bin/tesseract',  # Apple Silicon Homebrew
            '/usr/bin/tesseract'  # Linux default
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise Exception("Tesseract not found. Please install with: brew install tesseract")

    def extract_text(self, frame):
        """Extract text from an image frame using OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Perform OCR
        text = pytesseract.image_to_string(thresh, config=self.tess_config)
        return text.strip()

class ObjectTracker:
    def __init__(self):
        self.last_spoken_time = 0
        self.last_objects = {}

    def get_position(self, x_center, frame_width):
        pos = x_center / frame_width
        for name, (min_val, max_val) in POSITION_THRESHOLDS.items():
            if min_val <= pos < max_val:
                return name
        return "center"

    def generate_announcement(self, detections, frame_width):
        hazards = []
        normal = []
        position_changes = []
        
        for det in detections:
            obj_id = f"{det['label']}_{det['box'][0]}"
            x_center = (det["box"][0] + det["box"][2]) / 2
            current_pos = self.get_position(x_center, frame_width)
            
            if obj_id in self.last_objects:
                if current_pos != self.last_objects[obj_id]:
                    position_changes.append(f"{det['label']} moved to {current_pos}")
            
            self.last_objects[obj_id] = current_pos
            desc = f"{det['label']} {current_pos}"
            (hazards if det["label"] in HAZARD_OBJECTS else normal).append(desc)
        
        message = []
        if position_changes:
            message.append("Position change: " + ", ".join(position_changes))
        if hazards:
            message.append("Warning! " + ", ".join(hazards))
        if normal:
            message.append(", ".join(normal))
            
        return ". ".join(message) if message else ""

def main():
    # Initialize components
    speech_engine = SpeechEngine()
    ocr = OCRProcessor()
    tracker = ObjectTracker()
    cap = cv2.VideoCapture(0)
    text_mode = False
    running = True

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            # Immediate quit on Q
            if key == ord('q'):
                running = False
                break

            # Toggle modes
            if key == ord('t'):
                text_mode = not text_mode
                mode_text = "Text mode activated. Press space to read text." if text_mode else "Object detection mode activated."
                speech_engine.speak(mode_text)
                continue  # Skip processing this frame to avoid immediate detection

            if text_mode:
                # OCR Mode
                cv2.putText(frame, "TEXT MODE (SPACE=Read)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if key == ord(' '):
                    text = ocr.extract_text(frame)
                    if text:
                        speech_engine.speak(f"Text reads: {text}")
                    else:
                        speech_engine.speak("No text detected")
            else:
                # Object Detection Mode
                detections = detect(frame, conf_threshold=CONF_THRESHOLD)  # Now properly imported
                now = time.time()
                
                if detections and (now - tracker.last_spoken_time) >= SPEAK_INTERVAL:
                    announcement = tracker.generate_announcement(detections, frame.shape[1])
                    if announcement:
                        speech_engine.speak(announcement)
                        tracker.last_spoken_time = now

                # Visual feedback
                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    color = (0, 0, 255) if det["label"] in HAZARD_OBJECTS else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    position = tracker.get_position((x1 + x2)/2, frame.shape[1])
                    cv2.putText(frame, f"{det['label']} {position}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Vision Assistant (Q=Quit, T=Toggle Mode)", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_engine.stop()

if __name__ == "__main__":
    main()