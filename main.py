import cv2
import numpy as np
import argparse
import sys
from Process.pre_proc_image import pre_process_image
from Process.find_rotation import find_rotation
from Process.find_rotation_by_BB import find_rotation_by_BB
from Process.find_piece_center import find_piece_center
from Process.find_corners import find_corners

def main():
    parser = argparse.ArgumentParser(description="Piece Project CLI")
    parser.add_argument("-p", "--picture", help="Path to an image file to display", type=str)
    args = parser.parse_args()

    if args.picture:
        image = cv2.imread(args.picture)
        if image is None:
            print(f"Error: Could not load image from {args.picture}")
            sys.exit(1)
        
        # Resize to 500x500
        resized_image = cv2.resize(image, (500, 500))
        
        # Display image
        cv2.imshow("Original", resized_image)
        print(f"Displaying image from {args.picture} in 'Original' window (500x500)")

        pre_processed_image = pre_process_image(resized_image)
        cv2.imshow("Pre-processed image", pre_processed_image)

        # get center of mass of piece
        center_x, center_y = find_piece_center(pre_processed_image)

        rotation_angle, center_x, center_y, lines = find_rotation_by_BB(pre_processed_image, center_x, center_y)

        print(f"Rotation angle: {rotation_angle}  center of rotation: ({center_x}, {center_y})")
        
        # rotate the image around the point center_x, center_y by the rotation angle
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(pre_processed_image, rotation_matrix, (500, 500))
        
        cv2.imshow("Rotated Piece", rotated_image)
        print(f"Rotated image around ({center_x}, {center_y}) by {rotation_angle} degrees")
        
        # Find the corners of the piece
        corners = find_corners(rotated_image)
        cv2.imshow("Corners", corners)
        # print(f"Corners: {corners}")    
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Piece Project Initialized")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")

if __name__ == "__main__":
    main()
