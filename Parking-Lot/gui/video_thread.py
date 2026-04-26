from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import time
import numpy as np

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object)
    stats_signal = pyqtSignal(int, int, int)

    def __init__(self, manager, source=0, conf=0.30, overlap=0.15):
        super().__init__()
        self._run_flag = True
        self.source = source
        self.conf = conf
        self.overlap = overlap
        self.manager = manager
        self.detector = None
        self.lpr = None
        self.frame_count = 0
        self.calibrate_trigger = False

    def update_settings(self, source, conf, overlap):
        self.source = source
        self.conf = conf
        self.overlap = overlap
        # Note: Changing source requires restart usually, but we can handle conf/overlap dynamically

    def run(self):
        # Initialize models in the thread to avoid freezing GUI
        if self.detector is None:
            from core.detector import CarDetector
            self.detector = CarDetector()
        if self.lpr is None:
            from core.lpr_system import LPRSystem
            self.lpr = LPRSystem()

        # Capture from web cam
        cap = cv2.VideoCapture(self.source)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.frame_count += 1
                # Resize for consistent processing (matches GUI display)
                cv_img = cv2.resize(cv_img, (800, 450))
                
                # Check for calibration trigger
                if self.calibrate_trigger:
                    self.manager.calibrate(cv_img)
                    self.calibrate_trigger = False
                
                # Detect cars
                try:
                    detections = self.detector.detect(cv_img, conf=self.conf)
                except MemoryError:
                    print("Warning: Memory Error during detection. Skipping frame.")
                    continue
                except Exception as e:
                    print(f"Error during detection: {e}")
                    detections = []
                
                # Update occupancy
                # We need to pass overlap threshold if we want it to be dynamic, 
                # but currently check_occupancy doesn't take it. 
                # Let's assume we update the manager's threshold or pass it here.
                # For now, let's update manager's attribute if we can, or pass it.
                # Ideally, manager.check_occupancy(detections, overlap_threshold=self.overlap)
                self.manager.check_occupancy(cv_img, detections, overlap_threshold=self.overlap)
                
                # Run LPR every 30 frames on occupied spots
                if self.frame_count % 30 == 0:
                    for i, occupied in enumerate(self.manager.occupancy_status):
                        if occupied:
                            # Get spot coordinates
                            spot = self.manager.parking_spots[i]
                            # Crop the spot area
                            # Simple bounding box of the polygon
                            pts = np.array(spot, np.int32)
                            x, y, w, h = cv2.boundingRect(pts)
                            # Ensure within bounds
                            x, y = max(0, x), max(0, y)
                            
                            # Crop
                            spot_img = cv_img[y:y+h, x:x+w]
                            
                            # Read plate
                            if spot_img.size > 0:
                                text = self.lpr.read_plate(spot_img)
                                if text:
                                    self.manager.plates[i] = text
                        else:
                            self.manager.plates[i] = ""
                
                # Emit stats
                total, occupied, free = self.manager.get_stats()
                self.stats_signal.emit(total, occupied, free)
                
                self.change_pixmap_signal.emit(cv_img)
            else:
                # If reading from a file and it ends, loop or stop
                # For now, just stop or wait
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                # time.sleep(0.1)
            
            # Control frame rate roughly
            # time.sleep(0.03)
        
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
