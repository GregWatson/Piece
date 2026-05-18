import cv2
import numpy as np
import argparse
import sys
from Process.pre_proc_image import pre_process_image
from Piece_lib.find_rotation import find_rotation
from Piece_lib.find_piece_center import find_piece_center
from Process.find_corners import find_corners

def main():
    parser = argparse.ArgumentParser(description="Piece Project CLI")
    parser.add_argument("-p", "--picture", help="Path to an image file to display", type=str)
    parser.add_argument("-e", "--edges", help="Convert image specified by -p to an image of just edges, and save it to specified file.", type=str)
    args = parser.parse_args()

    if args.picture:
        image = cv2.imread(args.picture)
        if image is None:
            print(f"Error: Could not load image from {args.picture}")
            sys.exit(1)
        
        # Resize to 500x500
        resized_image = cv2.resize(image, (500, 500))
        
        # Display image
        if not args.edges:
            cv2.imshow("Original", resized_image)
            print(f"Displaying image from {args.picture} in 'Original' window (500x500)")

        # Clean up image to make it easier to analyze a jigsaw piece. This includes:
        # - Convert to grayscale
        # - Blur to reduce noise
        # - Threshold to get binary image
        # - Morphological operations to close gaps

        pre_processed_image = pre_process_image(resized_image)

        # Get the edges and lines, and find the rotation of the piece by finding the minimum bounding box of the lines. 
        # This is a rough approximation but should be good enough for rotation estimation.

        edges = cv2.Canny(pre_processed_image, 50, 150, apertureSize=3)

        if args.edges:
            cv2.imwrite(args.edges, edges)
            print(f"Saved edge-detected image to {args.edges}")
            return

        # get center of mass of piece
        center_x, center_y = find_piece_center(pre_processed_image)

        rotation_angle_rad, center_x, center_y, lines = find_rotation(edges, center_x, center_y)
        rotation_angle = np.degrees(rotation_angle_rad)

        print(f"Rotation angle: {rotation_angle}  center of rotation: ({center_x}, {center_y})")
        
        # rotate the image around the point center_x, center_y by the rotation angle
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(pre_processed_image, rotation_matrix, (500, 500))

        # draw the lines we found on the rotated image for debugging purposes           
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(rotated_image, (x1, y1), (x2, y2), (25, 155, 145), 1)

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
