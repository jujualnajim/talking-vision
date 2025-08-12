import cv2
from detection_api import detect
from ocr_processor import OCRProcessor
from text_to_speech import speak

def main():
    cap = cv2.VideoCapture(0)
    ocr = OCRProcessor()
    text_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        # Toggle modes
        if key == ord('t'):
            text_mode = not text_mode
            status = "TEXT MODE: ON" if text_mode else "OBJECT MODE: ON"
            speak(status)

        if text_mode:
            # OCR Mode
            cv2.putText(frame, "TEXT MODE (SPACE=Read)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if key == ord(' '):
                ocr.process_frame(frame)
        else:
            # Object Detection Mode
            detections = detect(frame)
            for obj in detections:
                label = obj["label"]
                speak(f"{label} detected")
                x1, y1, x2, y2 = obj["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Vision Assistant", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()