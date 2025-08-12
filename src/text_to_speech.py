import pyttsx3
import threading
from queue import Queue

class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()

    def _process_queue(self):
        while self.running:
            text = self.queue.get()
            if text is None:
                break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech Error: {e}")

    def speak(self, text):
        self.queue.put(text)

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join()

# Create a global instance
speech_engine = SpeechEngine()

# Define the speak function
def speak(text):
    speech_engine.speak(text)