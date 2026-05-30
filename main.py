import cv2
import numpy as np
import argparse
import sys
from Process.pre_proc_image import pre_process_image
from FL_lib.find_rotation import find_rotation
from FL_lib.fl_core import rotate_line, find_piece_center
from FL_lib.find_corners import find_corners

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

        # Extract edges using Canny edge detection. This will give us a binary image where the edges
        # are white and the rest is black. We can then use this to find lines and estimate rotation.
        edges = cv2.Canny(pre_processed_image, 50, 150, apertureSize=3)

        if args.edges:
            cv2.imwrite(args.edges, edges)
            print(f"Saved edge-detected image to {args.edges}")
            return

        # get center of mass of piece
        center_x, center_y = find_piece_center(pre_processed_image)

        rotation_angle_rad, center_x, center_y, lines = find_rotation(edges, center_x, center_y)
        rotation_angle = np.degrees(rotation_angle_rad)

        print(f"lines are {lines}")
        # print(f"Rotation angle: {rotation_angle}  center of rotation: ({center_x}, {center_y})")
        
        # rotate the image around the point center_x, center_y by the rotation angle
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        rotated_image_grey = cv2.warpAffine(pre_processed_image, rotation_matrix, (500, 500))
        rotated_image = cv2.cvtColor(rotated_image_grey, cv2.COLOR_GRAY2BGR)

        # Rotate the lines we found as well for debugging purposes, and draw them on the rotated image. 
        # This will help us see if the rotation is correct and if the lines are aligned with the edges 
        # of the piece after rotation. We can also use this to find the corners of the piece after rotation, 
        # which will be useful for further processing steps like matching pieces together.
        # We can also draw the lines we found on the rotated image for debugging purposes

        rotated_lines = [ rotate_line(line, (center_x, center_y), rotation_angle_rad) for line in lines ]

        # Show UN-rotated lines
        for (x1, y1), (x2, y2) in lines:
            cv2.line(rotated_image, (x1, y1), (x2, y2), (25, 155, 145), 1)

        # Show rotated lines
        for (x1,y1), (x2,y2) in rotated_lines:
            cv2.line(rotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (80, 255, 80), 3)

        cv2.imshow("Rotated Piece", rotated_image)
        print(f"Rotated image around ({center_x}, {center_y}) by {rotation_angle} degrees")
        
        # Find the corners of the piece
        corners = find_corners(rotated_lines, corner_thresh=50, end_to_end_dist_thresh=20, debug=True)
        for _, point, angle_rad in corners:
            cv2.circle(rotated_image, (int(point[0]), int(point[1])), 10, (0, 0, int(255*angle_rad/(2*np.pi))), -1)
        cv2.imshow("Corners", rotated_image)
        # print(f"Corners: {corners}")    
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Piece Project Initialized")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")

if __name__ == "__main__":
    main()
