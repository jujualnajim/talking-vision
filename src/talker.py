import time
import cv2
import pyttsx3
import detection_api
from collections import defaultdict

# Configuration
CONF_THRESHOLD = 0.5
SPEAK_INTERVAL = 3.0  # Fixed interval for repeating all objects
HAZARD_OBJECTS = ["stairs", "knife", "car", "fire"]
POSITION_THRESHOLDS = {
    "left": (0.0, 0.4),
    "center": (0.4, 0.6),
    "right": (0.6, 1.0)
}

class SpeechManager:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 200)  # Faster speech rate
        self.last_spoken_time = 0
        self.last_objects = set()
        
    def announce_objects(self, objects):
        """Speak all objects in a single announcement"""
        if not objects:
            return
            
        # Group hazards and normal objects
        hazards = []
        normal = []
        
        for label, position in objects:
            pos_desc = get_position_description(position)
            desc = f"{label} {pos_desc}"
            (hazards if label in HAZARD_OBJECTS else normal).append(desc)
        
        # Build announcement with hazards first
        announcement = []
        if hazards:
            announcement.append("Warning! " + ", ".join(hazards))
        if normal:
            announcement.append(", ".join(normal))
            
        full_announcement = ". ".join(announcement)
        
        try:
            # Stop any current speech
            self.engine.stop()
            # Start new announcement
            self.engine.say(full_announcement)
            self.engine.runAndWait()
            self.last_spoken_time = time.time()
            self.last_objects = objects
        except Exception as e:
            print(f"TTS error: {e}")

def get_position_description(position):
    for pos_name, (min_val, max_val) in POSITION_THRESHOLDS.items():
        if min_val <= position < max_val:
            return pos_name
    return "center"

def main():
    speech = SpeechManager()
    cap = cv2.VideoCapture(0)
    
    # For frame time calculation
    last_frame_time = time.time()
    frame_count = 0
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Running. Press Q to quit.")

    try:
        while True:
            # Start frame timer
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break

            # Detection (measure time for optimization)
            detect_start = time.time()
            detections = detection_api.detect(frame, conf_threshold=CONF_THRESHOLD)
            detect_time = time.time() - detect_start
            
            # Get current objects as immutable tuples
            current_objects = {
                (det["label"], det["position"]) 
                for det in detections
            }
            
            # Visual feedback
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                color = (0, 0, 255) if det["label"] in HAZARD_OBJECTS else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                position = get_position_description(det["position"])
                cv2.putText(frame, f"{det['label']} {det['conf']:.1f} {position}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Announcement logic
            now = time.time()
            if (now - speech.last_spoken_time) >= SPEAK_INTERVAL:
                speech.announce_objects(current_objects)
            
            # Display FPS (for debugging performance)
            frame_time = time.time() - frame_start
            fps = 1 / frame_time if frame_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f} | Detect: {detect_time*1000:.1f}ms",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Vision Assistant", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()