import cv2
import numpy as np
import pytesseract
import os
from text_to_speech import speak

class OCRProcessor:
    def __init__(self):
        # Try multiple possible Tesseract paths
        self.tesseract_cmd = self.find_tesseract()
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def find_tesseract(self):
        possible_paths = [
            '/usr/local/bin/tesseract',  # Homebrew default
            '/opt/homebrew/bin/tesseract',  # Apple Silicon Homebrew
            '/usr/bin/tesseract'  # Linux default
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise Exception("Tesseract not found. Please install it with 'brew install tesseract'")

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def extract_text(self, frame):
        try:
            processed = self.preprocess(frame)
            text = pytesseract.image_to_string(processed, config='--psm 6')
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def process_frame(self, frame):
        text = self.extract_text(frame)
        if text:
            speak(f"Text reads: {text}")
        else:
            speak("No text detected")