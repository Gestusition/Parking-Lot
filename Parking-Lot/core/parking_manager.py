import cv2
import numpy as np

class ParkingManager:
    def __init__(self):
        self.parking_spots = [] # List of polygons (list of points)
        self.occupancy_status = [] # Boolean list corresponding to spots
        self.plates = [] # List of strings corresponding to spots
        
        # Hysteresis counters
        self.occupancy_counter = [] 
        self.hysteresis_threshold = 5 # Frames required to change state
        
        from core.database import DatabaseManager
        self.db = DatabaseManager()
        
        # Visual Calibration
        self.reference_rois = [] # List of images (numpy arrays) representing empty spots

    def add_spot(self, points):
        """Add a parking spot defined by 4 points"""
        self.parking_spots.append(points)
        self.occupancy_status.append(False)
        self.plates.append("")
        self.occupancy_counter.append(0)

    def delete_spot(self, index):
        """Delete a parking spot by index"""
        if 0 <= index < len(self.parking_spots):
            self.parking_spots.pop(index)
            self.occupancy_status.pop(index)
            self.plates.pop(index)
            self.occupancy_counter.pop(index)
            if index < len(self.reference_rois):
                self.reference_rois.pop(index)

    def calibrate(self, frame):
        """
        Capture the current frame as the reference for empty spots.
        """
        self.reference_rois = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        for spot in self.parking_spots:
            # Extract ROI
            mask = np.zeros_like(gray)
            pts = np.array(spot, np.int32)
            cv2.fillPoly(mask, [pts], 255)
            # We store the masked spot area
            # Actually, simpler to just store the bounding rect crop for comparison
            # but masking is more precise.
            # Let's store the crop of the bounding rect, masked.
            x, y, w, h = cv2.boundingRect(pts)
            roi = gray[y:y+h, x:x+w].copy()
            mask_roi = mask[y:y+h, x:x+w]
            roi = cv2.bitwise_and(roi, roi, mask=mask_roi)
            self.reference_rois.append(roi)
        print("Calibration complete. Reference ROIs stored.")

    def check_occupancy(self, frame, detections, overlap_threshold=0.15):
        """
        Check which spots are occupied based on vehicle detections AND visual difference.
        frame: current video frame (BGR)
        detections: list of [x1, y1, x2, y2, conf, cls]
        overlap_threshold: minimum overlap ratio to consider occupied
        """
        # Store previous status to detect changes
        prev_status = self.occupancy_status.copy()
        
        # Ensure counters match spots length (in case of loading from json)
        if len(self.occupancy_counter) != len(self.parking_spots):
            self.occupancy_counter = [0] * len(self.parking_spots)

        # Create a snapshot of spots to avoid race conditions with add_spot
        spots = list(self.parking_spots)

        # Temporary status for this frame
        current_frame_status = [False] * len(spots)
        
        for i, spot in enumerate(spots):
            spot_poly = np.array(spot, np.int32)
            spot_area = cv2.contourArea(spot_poly)
            
            if spot_area == 0:
                continue

            for det in detections:
                x1, y1, x2, y2 = det[:4]
                
                # Create a polygon for the car bounding box
                car_poly = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], np.int32)
                
                # Calculate intersection
                intersection_area, _ = cv2.intersectConvexConvex(spot_poly, car_poly)
                
                # Check overlap ratio
                # We check if a significant portion of the spot is covered by a car
                # OR if a significant portion of a car is inside the spot
                
                # Ratio of spot covered by car
                spot_overlap_ratio = intersection_area / spot_area
                
                # Ratio of car inside spot (optional, but good for small cars in big spots)
                car_area = (x2 - x1) * (y2 - y1)
                car_overlap_ratio = 0
                if car_area > 0:
                    car_overlap_ratio = intersection_area / car_area
                
                # Thresholds (can be tuned)
                # If overlap_threshold of spot is covered OR 40% of car is inside spot
                if spot_overlap_ratio > overlap_threshold or car_overlap_ratio > 0.40:
                    current_frame_status[i] = True
                    break # Spot is occupied, move to next spot
            
            # Visual difference logic (Fallback if YOLO fails)
            if not current_frame_status[i] and len(self.reference_rois) == len(self.parking_spots):
                # Ensure we don't go out of bounds if spots changed
                if i >= len(self.reference_rois):
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                pts = np.array(spot, np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                
                # Ensure bounds
                if y+h <= gray.shape[0] and x+w <= gray.shape[1]:
                    roi = gray[y:y+h, x:x+w]
                    mask = np.zeros_like(roi)
                    pts_shifted = pts - np.array([x, y])
                    cv2.fillPoly(mask, [pts_shifted], 255)
                    
                    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
                    ref_masked = self.reference_rois[i]
                    
                    # Ensure shapes match (in case of resize or something, though unlikely if frame size const)
                    if roi_masked.shape == ref_masked.shape:
                        # Absolute difference
                        diff = cv2.absdiff(roi_masked, ref_masked)
                        # Mean of non-zero pixels (inside the spot)
                        # Or just mean of the whole rect (masked parts are 0 in both, so diff is 0)
                        # We need to divide by the area of the polygon, not the rect
                        score = np.sum(diff) / (cv2.countNonZero(mask) + 1)
                        
                        # Threshold for visual change (tune this)
                        # 25-30 is usually a good starting point for significant change
                        if score > 25: 
                            # Check for "3D" structure change using edges
                            # Shadows (2D) change intensity but often preserve underlying edges (like grid lines)
                            # Objects (3D) occlude background edges and add new ones.
                            
                            # Calculate edges for current ROI and reference ROI
                            # Lower thresholds to detect faint edges (like grid lines or car contours)
                            edges_roi = cv2.Canny(roi_masked, 30, 100)
                            edges_ref = cv2.Canny(ref_masked, 30, 100)
                            
                            # Mask edges to ensure we only look at the spot area
                            edges_roi = cv2.bitwise_and(edges_roi, edges_roi, mask=mask)
                            edges_ref = cv2.bitwise_and(edges_ref, edges_ref, mask=mask)
                            
                            # Calculate difference in edges
                            edge_diff = cv2.absdiff(edges_roi, edges_ref)
                            # Normalize edge score
                            edge_score = np.sum(edge_diff) / (cv2.countNonZero(mask) + 1)
                            
                            # If edge score is low, it's likely just a shadow/lighting change (2D)
                            # If it's a car, edge_score should be high (3D structure change)
                            # Lowered threshold to 5 to be more sensitive to car presence
                            # Added intensity override: if intensity change is MASSIVE (>60), it's likely an object
                            if edge_score > 5 or score > 60: 
                                current_frame_status[i] = True
                            else:
                                # print(f"Spot {i}: Intensity change ({score:.1f}) but low edge diff ({edge_score:.1f}) -> Shadow ignored")
                                pass
        
        # Apply Hysteresis
        for i, is_occupied_now in enumerate(current_frame_status):
            if i >= len(self.occupancy_counter) or i >= len(self.occupancy_status):
                continue

            if is_occupied_now:
                # Increment counter if car detected
                self.occupancy_counter[i] = min(self.occupancy_counter[i] + 1, self.hysteresis_threshold + 2)
            else:
                # Decrement counter if no car
                self.occupancy_counter[i] = max(self.occupancy_counter[i] - 1, 0)
            
            # Update status based on counter
            if self.occupancy_counter[i] >= self.hysteresis_threshold:
                self.occupancy_status[i] = True
            elif self.occupancy_counter[i] == 0:
                self.occupancy_status[i] = False
        
        # Check for changes and log (compare against the smoothed status)
        for i, occupied in enumerate(self.occupancy_status):
            if i >= len(prev_status):
                continue
                
            if occupied and not prev_status[i]:
                # Entry event
                self.db.log_event(i, "Unknown", "ENTRY")
            elif not occupied and prev_status[i]:
                # Exit event
                plate = self.plates[i] if i < len(self.plates) and self.plates[i] else "Unknown"
                self.db.log_event(i, plate, "EXIT")
                if i < len(self.plates):
                    self.plates[i] = "" # Clear plate on exit

    def get_stats(self):
        total = len(self.parking_spots)
        occupied = sum(self.occupancy_status)
        free = total - occupied
        return total, occupied, free

    def save_to_json(self, filename='parking_spots.json'):
        import json
        with open(filename, 'w') as f:
            json.dump(self.parking_spots, f)

    def load_from_json(self, filename='parking_spots.json'):
        import json
        import os
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.parking_spots = json.load(f)
                self.occupancy_status = [False] * len(self.parking_spots)
                self.plates = [""] * len(self.parking_spots)
                self.occupancy_counter = [0] * len(self.parking_spots)

