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
from find_tabs import find_tabs
from find_blanks import find_blank_lines
from find_triangles import find_triangles_from_corners

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
        pre_processed_image = fl_remove_background(resized_image, debug=args.debug, image_type=args.type)
        show_image(pre_processed_image, str="Pre-processed", max=1000, wait_for_key=True)

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
            # if idx != 23 : continue
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

            rotation_angle_rad, lines = find_rotation(edges, cx, cy, debug=args.debug)
            rotation_angle = np.degrees(rotation_angle_rad)

            # rotate the image around the point cx, cy by the rotation angle
            rotation_matrix = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
            rotated_image_grey = cv2.warpAffine(piece_image, rotation_matrix, (500, 500))
            rotated_image = cv2.cvtColor(rotated_image_grey, cv2.COLOR_GRAY2BGR)
            w, h = rotated_image_grey.shape[1], rotated_image_grey.shape[0]
            rotated_edges = edges.copy()
            rotated_edges = cv2.warpAffine(rotated_edges, rotation_matrix, (w, h))

            # Rotate the lines we found as well for debugging purposes, and draw them on the rotated image. 
            # This will help us see if the rotation is correct and if the lines are aligned with the edges 
            # of the piece after rotation. We can also use this to find the corners of the piece after rotation, 
            # which will be useful for further processing steps like matching pieces together.
            # We can also draw the lines we found on the rotated image for debugging purposes

            rotated_lines = [ rotate_line(line, (cx, cy), rotation_angle_rad) for line in lines ]
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
            
            tab_lines = find_tabs(rotated_lines, ((tl_x, tl_y), (br_x, br_y)))
            blank_lines = find_blank_lines(rotated_lines, ((tl_x, tl_y), (br_x, br_y)), (cx,cy))

            # Find the corners of the piece
            corners = find_corners(rotated_lines, (tl_x, tl_y), (br_x, br_y), tab_lines=tab_lines, end_to_end_dist_thresh=20, debug=args.debug)
            # for _, point, angle_rad in corners:
            #     cv2.circle(rotated_image, (int(point[0]), int(point[1])), 10, (0, 0, int(255*angle_rad/(2*np.pi))), -1)
            # show_image(rotated_image, str=f"Corners {idx+1}", max=1000, wait_for_key=True)
            # print(f"Corners: {corners}")

            # Get triangles formed by corners and the point furthest from the line between 2 corners.
            # Operate on rotated edge image.
            triangles = find_triangles_from_corners(rotated_edges, corners, debug=args.debug)

            unrotated_corner_points = []
            for row in corners:
                unrotated_row = []
                for corner in row:
                    if corner:
                        _, point, _ = corner
                        unrotated_point = rotate_point(point, (cx, cy), -rotation_angle_rad)
                        pt_in_orig = inverse_transform_fn(map(int, unrotated_point))
                        unrotated_row.append( (int(pt_in_orig[0]), int(pt_in_orig[1])) )
                    else:
                        unrotated_row.append(False)
                unrotated_corner_points.append(unrotated_row)

            # Also add corners to the original image.
            for corner_point in [ col for row in unrotated_corner_points for col in row if col]:
                cv2.circle(resized_image, corner_point, 10, (0, 0, 255), -1)

            # Also add tab points to original image
            for line in tab_lines:
                unrotated_line = rotate_line(line, (cx, cy), -rotation_angle_rad)
                o_pts = [inverse_transform_fn(map(int, pt)) for pt in unrotated_line]
                cv2.line(resized_image, (int(o_pts[0][0]), int(o_pts[0][1])), (int(o_pts[1][0]), int(o_pts[1][1])), (250, 0,0), 4)

            # Also add blanks to original image
            for line in blank_lines:
                unrotated_line = rotate_line(line, (cx, cy), -rotation_angle_rad)
                o_pts = [inverse_transform_fn(map(int, pt)) for pt in unrotated_line]
                cv2.line(resized_image, (int(o_pts[0][0]), int(o_pts[0][1])), (int(o_pts[1][0]), int(o_pts[1][1])), (250, 0, 200), 5)

            # Put number on piece
            cv2.putText(resized_image, f"{idx}", (piece['centroid'][0] - 20, piece['centroid'][1] + 20) , cv2.FONT_HERSHEY_SIMPLEX, 3.0, (100, 100, 100), 5)

            # Show centroid
            cv2.circle(resized_image, (int(piece['centroid'][0]), int(piece['centroid'][1])), 5, (200,30,30), 6)

        show_image(resized_image, "Orig with corners.", max=1000, wait_for_key=True)
        cv2.destroyAllWindows()
    else:
        print("Piece Project Initialized")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")

if __name__ == "__main__":
    main()
