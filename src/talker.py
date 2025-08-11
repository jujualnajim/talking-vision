import time
import cv2
import pyttsx3
import detection_api

CONF_THRESHOLD = 0.5
SPEAK_COOLDOWN_SEC = 1.5

def init_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    return engine

def speak(engine, text):
    if not text:
        return
    try:
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

def main():
    engine = init_tts()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    speak_enabled = True
    last_spoken = ""
    last_spoken_at = 0.0

    print("Running. Keys: [S]=toggle speech, [Q]=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detection_api.detect(frame, conf_threshold=CONF_THRESHOLD)

            top_label = None
            if detections:
                top = max(detections, key=lambda d: d["conf"])
                top_label = top["label"]

            now = time.time()
            if speak_enabled and top_label:
                cooled = (now - last_spoken_at) >= SPEAK_COOLDOWN_SEC
                if cooled:
                    speak(engine, top_label)
                    last_spoken = top_label
                    last_spoken_at = now

            cv2.imshow("Talking Vision (S=speech, Q=quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                speak_enabled = not speak_enabled
                print(f"Speech: {'ON' if speak_enabled else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
