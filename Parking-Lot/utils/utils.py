import cv2
import numpy as np

def draw_parking_spots(image, spots, status, plates=None):
    """
    Draws parking spots on the image.
    spots: list of polygons
    status: list of booleans (True=Occupied, False=Free)
    plates: list of strings (optional)
    """
    overlay = image.copy()
    
    for i, spot in enumerate(spots):
        points = np.array(spot, np.int32)
        points = points.reshape((-1, 1, 2))
        
        color = (0, 0, 255) if status[i] else (0, 255, 0) # Red if occupied, Green if free
        
        cv2.polylines(overlay, [points], True, color, 2)
        cv2.fillPoly(overlay, [points], color)
        
        # Draw plate text if available
        if plates and i < len(plates) and plates[i]:
            # Calculate center
            M = cv2.moments(points)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(image, plates[i], (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    # Apply transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image
