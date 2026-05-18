# Find the center of mass of the white pixels in the pre-processed image.
# (Image must be purely black and white, with the piece in white and the background in black.)

import numpy as np

def find_piece_center(img):
    # Find the coordinates of all white pixels
    white_pixels = np.argwhere(img == 255)
    
    # Calculate the center of mass
    center_x = np.mean(white_pixels[:, 1])
    center_y = np.mean(white_pixels[:, 0])
    
    return int(center_x), int(center_y)
