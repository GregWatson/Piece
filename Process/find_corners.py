import cv2
import numpy as np
import itertools

def find_corners(img):
    # Find a larger pool of potential corners
    # goodFeaturesToTrack is more robust for finding a specific number of corners
    candidates = cv2.goodFeaturesToTrack(img, maxCorners=10, qualityLevel=0.01, minDistance=50, useHarrisDetector=True)
    
    # Create a copy of the image to mark the corners (convert to BGR for color drawing)
    marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if candidates is not None and len(candidates) >= 4:
        # Reshape candidates for easier manipulation
        pts = [tuple(p.ravel()) for p in candidates]
        
        max_area = 0
        best_corners = None
        
        # Test all combinations of 4 points to find the one with the largest area
        for combo in itertools.combinations(pts, 4):
            # Calculate the area of the quadrilateral formed by these 4 points
            # We use convexHull to ensure points are in order for contourArea
            hull = cv2.convexHull(np.array(combo, dtype=np.int32))
            area = cv2.contourArea(hull)
            
            if area > max_area:
                max_area = area
                best_corners = combo
        
        if best_corners:
            for pt in best_corners:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(marked_img, (x, y), 8, (0, 0, 255), -1) # Draw red circle
    elif candidates is not None:
        # If fewer than 4 candidates, just mark what we found
        for i in candidates:
            x, y = i.ravel()
            cv2.circle(marked_img, (int(x), int(y)), 5, (0, 0, 255), -1)
    else:
        print("No corners found")

    return marked_img
