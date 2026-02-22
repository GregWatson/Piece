import cv2
import numpy as np

def find_rotation(img):
    if img is None:
        print("Error: Image is None")
        return 0

    gray = img

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Transform (Probabilistic)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=20 , maxLineGap=10)

    angle = 0
    line_count = 0
    
    if lines is not None:
        line_count = len(lines)
        angles = []
        
        # Create a copy for drawing
        display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate angle in degrees
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            
            # Normalize angle to [-45, 45] or similar for rotation correction
            if angle_deg > 45:
                angle_deg -= 90
            elif angle_deg < -45:
                angle_deg += 90
            
            angles.append(angle_deg)

        if angles:
            angle = np.median(angles)

        cv2.imshow("Detected Lines", display_img)

    print(f"Lines found: {line_count}")
    return angle
