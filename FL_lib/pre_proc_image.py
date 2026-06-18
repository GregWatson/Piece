import cv2
import numpy as np

from FL_lib.fl_core import show_image

def pre_process_image(img, debug=False):
    """
    Clean up image to make it easier to analyze a jigsaw piece. This includes:
    - Convert to grayscale
    - Blur to reduce noise
    - Threshold to get binary image
    - Morphological operations to close gaps
    """
    # - Convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    # Assume background is either very dark or very light compared to pieces.
    # We can try OTSU or adaptive. User said "solid background".
    # Let's try Otsu first as it's robust for bimodal histograms.
    opt_thresh, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # print(f"Optimum threshold: {opt_thresh}")

    # Invert if the background is detected as white (pieces are black)
    # Usually we want pieces to be white (255) and background black (0) for findContours
    # Simple check: if corners are white, inverted.
    h, w = thresh.shape
    corners = [thresh[0,0], thresh[0, w-1], thresh[h-1, 0], thresh[h-1, w-1]]
    if sum(corners) / 4 > 127: 
        thresh = cv2.bitwise_not(thresh)
        
    # Morphological operations to close gaps
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return thresh