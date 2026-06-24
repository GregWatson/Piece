from email.mime import image

import cv2
import numpy as np
import argparse
import sys
import os

# Get the absolute path to the directory containing the module
module_path = os.path.abspath("../FL_lib")

# Add it to the Python search path
if module_path not in sys.path:
    # print(f"Adding path {module_path} to sys.path")
    sys.path.append(module_path)
    
from pre_proc_image import pre_process_image
from find_rotation import find_rotation
from fl_core import rotate_line, show_image, get_bounding_box_from_lines, rotate_point
from find_corners import find_corners
from get_piece_info import get_piece_info
from fl_remove_background import fl_remove_background
from fl_pad_and_scale import fl_pad_and_scale

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
        # show_image(cleaned_up, str="Background removed", max=1000, wait_for_key=True)
        # exit(0)

        # - Convert to grayscale
        # - Blur to reduce noise
        # - Threshold to get binary image
        # - Morphological operations to close gaps
        pre_processed_image = pre_process_image(cleaned_up, debug=args.debug)

        # Find the number of pieces in the image using the get_piece_info function
        pieces = get_piece_info(pre_processed_image)

        pieces_img = pre_processed_image.copy()
        print(f"Detected {len(pieces)} piece(s) in the image.")
        for idx, piece in enumerate(pieces):
            print(f"Piece {idx+1}: Bounding Box = {piece['box']}, Centroid = {piece['centroid']}, Area = {piece['area']}")
            # draw bbox and centroid in grey
            x,y,w,h = piece['box']
            cv2.rectangle(pieces_img, (x,y), (x+w, y+h), (127,127,127),2)
            cv2.circle(pieces_img, (int(piece['centroid'][0]), int(piece['centroid'][1])), 5, (127,127,127), -1)
        show_image(pieces_img, str="Pre-processed Image with bbox and centroids", max=1000, wait_for_key=True)


        # for each piece, we want to find the edges and lines and corners.
        for idx, piece in enumerate(pieces):
            # if idx != 8 : continue
            print(f"\nProcessing Piece {idx+1}: Bounding Box = {piece['box']}, Centroid = {piece['centroid']}, Area = {piece['area']}")

            # create a new image that is just the piece, by cropping the pre_processed_image using the 
            # bounding box of the piece.
            orig_x, orig_y, w, h = piece['box']
            cx, cy = piece['centroid']
            piece_image, _, rot_center, inverse_transform_fn = fl_pad_and_scale(pre_processed_image, 
                                                                                [[(orig_x,orig_y), (orig_x+w, orig_y+h)]],
                                                                                piece['centroid'],
                                                                                new_img_size = 500, 
                                                                                debug=args.debug)

            cx,cy = rot_center

            # Extract edges using Canny edge detection. This will give us a binary image where the edges
            # are white and the rest is black. We can then use this to find lines and estimate rotation.
            edges = cv2.Canny(piece_image, 50, 150, apertureSize=3)

            if args.edges:
                cv2.imwrite(args.edges, edges)
                print(f"Saved edge-detected image to {args.edges}")
                return

            rotation_angle_rad, center_x, center_y, lines = find_rotation(edges, cx, cy)
            rotation_angle = np.degrees(rotation_angle_rad)

            # rotate the image around the point center_x, center_y by the rotation angle
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
            rotated_image_grey = cv2.warpAffine(piece_image, rotation_matrix, (500, 500))
            rotated_image = cv2.cvtColor(rotated_image_grey, cv2.COLOR_GRAY2BGR)
            w, h = rotated_image_grey.shape[1], rotated_image_grey.shape[0]

            # Rotate the lines we found as well for debugging purposes, and draw them on the rotated image. 
            # This will help us see if the rotation is correct and if the lines are aligned with the edges 
            # of the piece after rotation. We can also use this to find the corners of the piece after rotation, 
            # which will be useful for further processing steps like matching pieces together.
            # We can also draw the lines we found on the rotated image for debugging purposes

            rotated_lines = [ rotate_line(line, (center_x, center_y), rotation_angle_rad) for line in lines ]
            # get top_left and bottom right bounding box of rotated lines.
            (tl_x, tl_y), (br_x, br_y) = get_bounding_box_from_lines(rotated_lines)

            # draw bbox of lines on rotated image for debugging purposes
            cv2.rectangle(rotated_image, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 1)

            # Show UN-rotated lines
            for (x1, y1), (x2, y2) in lines:
                cv2.line(rotated_image, (x1, y1), (x2, y2), (25, 155, 145), 1)

            # Show rotated lines
            for (x1,y1), (x2,y2) in rotated_lines:
                cv2.line(rotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (80, 255, 80), 3)
            
            # Find the corners of the piece
            corners = find_corners(rotated_lines, (tl_x, tl_y), (br_x, br_y), end_to_end_dist_thresh=20, debug=args.debug)
            # for _, point, angle_rad in corners:
            #     cv2.circle(rotated_image, (int(point[0]), int(point[1])), 10, (0, 0, int(255*angle_rad/(2*np.pi))), -1)
            # show_image(rotated_image, str=f"Corners {idx+1}", max=1000, wait_for_key=True)
            # print(f"Corners: {corners}")    

            # Also add corners to the original image.
            for _, point, _ in corners:
                unrotated_point = rotate_point(point, (center_x, center_y), -rotation_angle_rad)
                pt_in_orig = inverse_transform_fn(map(int, unrotated_point))
                cv2.circle(resized_image, (int(pt_in_orig[0]), int(pt_in_orig[1])), 10, (0, 0, 255), -1)

            # add corners to the original resized image

        show_image(resized_image, "Orig with corners.", max=1000, wait_for_key=True)
        cv2.destroyAllWindows()
    else:
        print("Piece Project Initialized")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")

if __name__ == "__main__":
    main()
