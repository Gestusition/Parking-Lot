from ultralytics import YOLO
import cv2

class CarDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        # Car, truck, bus, motorcycle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7] 

    def detect(self, frame):
        """
        Detects vehicles in the frame.
        Returns list of bounding boxes [x1, y1, x2, y2, conf, cls]
        """
        results = self.model(frame, verbose=False, conf=0.1)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    detections.append([int(x1), int(y1), int(x2), int(y2), conf, cls])
                    
        return detections
