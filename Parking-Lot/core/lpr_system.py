import cv2
import numpy as np

class LPRSystem:
    def __init__(self):
        import easyocr
        # Initialize EasyOCR reader for English (can add others)
        # Force CPU to avoid delays
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read_plate(self, image_crop):
        """
        Reads text from a cropped image of a license plate.
        """
        result = self.reader.readtext(image_crop)
        # Filter results, maybe look for specific patterns
        text = ""
        for (bbox, t, prob) in result:
            if prob > 0.5:
                text += t + " "
        return text.strip()
