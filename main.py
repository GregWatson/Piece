from email.mime import image

import cv2
import numpy as np
import argparse
import sys
from FL_lib.pre_proc_image import pre_process_image
from FL_lib.find_rotation import find_rotation
from FL_lib.fl_core import rotate_line, get_distance_between_2_points, show_image
from FL_lib.find_corners import find_corners
from FL_lib.get_piece_info import get_piece_info
from FL_lib.fl_remove_background import fl_remove_background

def main():
    parser = argparse.ArgumentParser(description="Piece Project CLI")
    parser.add_argument("-p", "--picture", help="Path to an image file to display", type=str)
    parser.add_argument("-t", "--type", help="Type of image: normal = one or more jigsaw pieces. reverse = the reverse side of the pieces (typically blank). Default is normal", choices=["normal", "reverse"], default="normal")
    parser.add_argument("-e", "--edges", help="Convert image specified by -p to an image of just edges, and save it to specified file.", type=str)
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with verbose output")
    args = parser.parse_args()

    if args.picture:
        image = cv2.imread(args.picture)
        if image is None:
            print(f"Error: Could not load image from {args.picture}")
            sys.exit(1)
        
        # if the image is less than 500 x 500 then enlarge it to 500 x 500 for better processing. 
        # We can use cv2.resize for this, and we can use interpolation to maintain quality. This will help us ensure that the line detection and rotation estimation works well even for smaller images.
        # scale_factor = max(1.0, 500.0 / max(image.shape[0], image.shape[1]))
        resized_image = image.copy()
        # resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        # print(f"Loaded image from {args.picture} with original size {image.shape[1]}x{image.shape[0]}, resized to {resized_image.shape[1]}x{resized_image.shape[0]} for processing.")
        
        # Clean up image to make it easier to analyze a jigsaw piece.
        cleaned_up = fl_remove_background(resized_image, debug=args.debug, image_type=args.type)
        show_image(cleaned_up, str="Background removed", max=1000, wait_for_key=True)
        # exit(0)

        # - Convert to grayscale
        # - Blur to reduce noise
        # - Threshold to get binary image
        # - Morphological operations to close gaps


        pre_processed_image = pre_process_image(cleaned_up, debug=args.debug)
        show_image(pre_processed_image, str="Pre-processed Image", max=1000, wait_for_key=True)

        # Find the number of pieces in the image using the get_piece_info function
        pieces = get_piece_info(pre_processed_image)
        print(f"Detected {len(pieces)} piece(s) in the image.")
        for idx, piece in enumerate(pieces):
            print(f"Piece {idx+1}: Bounding Box = {piece['box']}, Centroid = {piece['centroid']}, Area = {piece['area']}")

        # for each piece, we want to find the edges and lines and corners.
        for idx, piece in enumerate(pieces):
            print(f"\nProcessing Piece {idx+1}: Bounding Box = {piece['box']}, Centroid = {piece['centroid']}, Area = {piece['area']}")

            # create a new image that is just the piece, by cropping the pre_processed_image using the 
            # bounding box of the piece.
            x, y, w, h = piece['box']
            piece_image = pre_processed_image[y:y+h, x:x+w]

            # compute the radius of the surrounding circle.
            # We can use this to increase the canvas size to ensure that we can rotate the piece without cutting off any parts of it. 
            diameter = get_distance_between_2_points((x,y), (x+w, y+h))

            add_height = max(0, int(diameter - h)) //2 + 10
            add_width = max(0, int(diameter - w)) // 2 + 10

            # Add black padding around the piece to make it 500x500 so that the rotation and line detection works better. We can add a padding of 50 pixels on each side, which will give us a 600x600 image. This will also help us avoid issues with pieces that are close to the edge of the image.
            piece_image = cv2.copyMakeBorder(piece_image, add_height, add_height, add_width, add_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # scale the image to 500 x 500 for better processing. We can use cv2.resize for this, and we can use interpolation to maintain quality. This will help us ensure that the line detection and rotation estimation works well even for smaller pieces.
            scale_factor = 500.0 / max(piece_image.shape[0], piece_image.shape[1])
            piece_image = cv2.resize(piece_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

            # display the piece image in a new window for debugging purposes
            if args.debug: cv2.imshow(f"Piece {idx+1}", piece_image)
            
            # Get the edges and lines, and find the rotation of the piece by finding the minimum bounding box of the lines. 
            # This is a rough approximation but should be good enough for rotation estimation.

            # Extract edges using Canny edge detection. This will give us a binary image where the edges
            # are white and the rest is black. We can then use this to find lines and estimate rotation.
            edges = cv2.Canny(piece_image, 50, 150, apertureSize=3)

            if args.edges:
                cv2.imwrite(args.edges, edges)
                print(f"Saved edge-detected image to {args.edges}")
                return

            # get center of image
            (center_x, center_y) = (piece_image.shape[1] // 2, piece_image.shape[0] // 2)

            rotation_angle_rad, center_x, center_y, lines = find_rotation(edges, center_x, center_y)
            rotation_angle = np.degrees(rotation_angle_rad)

            # rotate the image around the point center_x, center_y by the rotation angle
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
            rotated_image_grey = cv2.warpAffine(piece_image, rotation_matrix, (500, 500))
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

            #cv2.imshow("Rotated Piece", rotated_image)
            #print(f"Rotated image around ({center_x}, {center_y}) by {rotation_angle} degrees")
            
            # Find the corners of the piece
            corners = find_corners(rotated_lines, corner_thresh=50, end_to_end_dist_thresh=20, debug=False)
            for _, point, angle_rad in corners:
                cv2.circle(rotated_image, (int(point[0]), int(point[1])), 10, (0, 0, int(255*angle_rad/(2*np.pi))), -1)
            cv2.imshow(f"Corners {idx+1}", rotated_image)
            # print(f"Corners: {corners}")    
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Piece Project Initialized")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")

if __name__ == "__main__":
    main()
