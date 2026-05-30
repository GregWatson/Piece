import numpy as np
import cv2

# Rotate a cv2 image by a specified amount (radians) around a given point
def rotate_image(image, rotation_angle_rad, center_x, center_y, sizex=None, sizey=None):

    if sizex is None:
        sizex = image.shape[1]
    if sizey is None:
        sizey = image.shape[0]

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(rotation_angle_rad), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (sizex, sizey))

    return rotated_image