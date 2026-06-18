import cv2
import numpy as np
from rembg import remove
from FL_lib.fl_core import show_image

# given a BGR image, remove the background and return a cleaned up image with just the pieces. 
# We can use the REMBG library to do this, which is a pre-trained model for background removal. 
# We can then apply a mask to the original image to get the cleaned up image. 
# This will help us ensure that the line detection and rotation estimation works well even for smaller images.
def fl_remove_background(img, debug=False, image_type="normal"):

    if (image_type == "reverse"):
        # For reverse images, we can try a simpler approach since the pieces are typically blank and the background is solid. 
        # We can convert to grayscale and then apply a threshold to create a binary mask. 
        # This will be faster than using REMBG and should work well for reverse images.

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use Otsu's thresholding to automatically determine the best threshold value
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert the thresholded image if the pieces are detected as black (0) and background as white (255)
        if np.sum(thresh == 255) > np.sum(thresh == 0):
            thresh = cv2.bitwise_not(thresh)

        # Use the thresholded image as a mask to extract the pieces from the original image
        clean = cv2.bitwise_and(img, img, mask=thresh)

        if debug:
            show_image(thresh, str="Threshold Mask", max=1000, wait_for_key=True)
            show_image(clean, str="Cleaned Image", max=1000, wait_for_key=True)

        return clean
    
    # Images of the normal type can be more complex, with pieces that have patterns and colors, 
    # and backgrounds that may not be perfectly solid.
    # Use the REMBG lib to remove the background and return a mask.
    # Mask is returns an array (entry per pixel) of a 4-tuple: BGR and alpha. 
    # The alpha channel is 0 for background and 255 for foreground (pieces).
    mask = remove(img, mask=True)
    print(f"mask shape: {mask.shape}, mask dtype: {mask.dtype}, unique values in mask: {np.unique(mask)}")

    # convert the alpha channel to a greyscale image and save it to disk for debugging.
    mask_as_grey = mask[:, :, 3]  # alpha channel
    # cv2.imwrite("mask.png", mask_as_grey)
    show_image(mask_as_grey, str="Mask Alpha Channel", max=1000, wait_for_key=True)

    if debug:
        show_image(img, str="Background removed", max=1000, wait_for_key=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. Create a mask where the sum is greater than 10
    mask = gray > 10

    clean = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8)*255)

    return clean