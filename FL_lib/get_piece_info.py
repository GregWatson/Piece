import cv2
import numpy as np

def get_piece_info(pre_processed_image, min_area=100, debug=False):
    """
    Analyzes a pre-processed binary image (where puzzle pieces are white foreground (255)
    and the background is black (0)) to detect individual puzzle pieces.

    Parameters:
        pre_processed_image (numpy.ndarray): Binary image containing one or more puzzle pieces.
        min_area (int or float): The minimum contour area to consider as a valid puzzle piece,
                                 filtering out small noise contours. Defaults to 100.

    Returns:
        list of dict: A list containing information for each detected puzzle piece.
                      Each piece's dictionary has the following keys:
                      - 'box': Tuple of (x, y, w, h) representing the upright bounding box.
                      - 'contour': The contour array found by OpenCV.
                      - 'area': The float area of the contour.
                      - 'centroid': Tuple of (cx, cy) representing the center of mass of the piece.
    """
    # Find contours
    # cv2.RETR_EXTERNAL retrieves only the outermost contours
    contours, _ = cv2.findContours(pre_processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pieces = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Get upright bounding box (x, y, w, h)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate centroid using image moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback to bounding box center if moment area is zero
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            
        piece_info = {
            'box': (x, y, w, h),
            'contour': contour,
            'area': area,
            'centroid': (cx, cy)
        }
        
        pieces.append(piece_info)
        
        if debug:
            print(f"Detected piece with area {area}, bounding box {x, y, w, h}, centroid {cx, cy}")
    return pieces
