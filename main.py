import cv2
import numpy as np
import argparse
import sys
import os
import re


# Get the absolute path to the directory containing the module
module_path = os.path.abspath("../FL_lib")

# Add it to the Python search path
if module_path not in sys.path:
    # print(f"Adding path {module_path} to sys.path")
    sys.path.append(module_path)
    
# from pre_proc_image import pre_process_image
from fl_types import J_Piece, P_Info, Jigsaw
from find_rotation import find_rotation
from fl_core import rotate_line, show_image, get_bounding_box_from_lines, rotate_point, draw_poly
from fl_core import rotate_and_transform_point, draw_triangle
from find_corners import find_corners
from get_piece_info import get_piece_info
from fl_remove_background import fl_remove_background
from fl_pad_and_scale import fl_pad_and_scale
from find_triangles import find_triangles_from_corners
from process_piece import process_piece
from solve_puzzle import solve_puzzle

def main():
    parser = argparse.ArgumentParser(description="Piece Project CLI")
    parser.add_argument("-p", "--picture", help="Path to an image file to display", type=str)
    parser.add_argument("-t", "--type", help="Type of image: normal = one or more jigsaw pieces. reverse = the reverse side of the pieces (typically blank). Default is normal", choices=["normal", "reverse"], default="normal")
    parser.add_argument("-e", "--edges", help="Convert image specified by -p to an image of just edges, and save it to specified file.", type=str)
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("-s", "--size", help="Specify jigsaw size in pieces: WxH. Default 6x6", type=str, default='6x6')

    args = parser.parse_args()

    
    if args.picture:
        image = cv2.imread(args.picture)
        if image is None:
            print(f"Error: Could not load image from {args.picture}")
            sys.exit(1)
        
        # if the image is less than 500 x 500 then enlarge it to 500 x 500 for better processing. 
        # We can use cv2.resize for this, and we can use interpolation to maintain quality. This will help us ensure that the line detection and rotation estimation works well even for smaller images.
        scale_factor = max(1.0, 2000.0 / max(image.shape[0], image.shape[1]))
        #resized_image = image.copy()
        resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        print(f"Loaded image from {args.picture} with original size {image.shape[1]}x{image.shape[0]}, resized to {resized_image.shape[1]}x{resized_image.shape[0]} for processing.")
        
        # Clean up image to make it easier to analyze a jigsaw piece.
        pre_processed_image = fl_remove_background(resized_image, debug=args.debug, image_type=args.type)
        show_image(pre_processed_image, str="Pre-processed", max_side=1000, wait_for_key=True)

        # Find basic info on each piece in the image using the get_piece_info function.
        piece_info: list [P_Info] = get_piece_info(pre_processed_image)
        pieces = [ J_Piece(info=info) for info in piece_info ]

        print(f"Detected {len(pieces)} piece(s) in the image.")

        # pieces_img = pre_processed_image.copy()
        # for idx, piece in enumerate(pieces):
        #     print(f"Piece {idx+1}: Bounding Box = {piece['box']}, Centroid = {piece['centroid']}, Area = {piece['area']}")
        #     # draw bbox and centroid in grey
        #     x,y,w,h = piece['box']
        #     cv2.rectangle(pieces_img, (x,y), (x+w, y+h), (127,127,127),2)
        #     cv2.circle(pieces_img, (int(piece['centroid'][0]), int(piece['centroid'][1])), 5, (127,127,127), -1)
        # show_image(pieces_img, str="Pre-processed Image with bbox and centroids", max_side=1000, wait_for_key=True)


        # for each piece, we want to find the edges and lines and corners.
        for piece in pieces:
            info = piece.info
            idx = info.id
            # if idx != 1: continue

            print(f"\nProcessing Piece number {info.id+1} (ID is {info.id} ): Bounding Box = {info.box}, Centroid = {info.centroid}, Area = {info.area}")

            # process_piece does the heavy lifting to analyze each piece and extract import information
            # such as edge types, lines, and corner points.

            ok = process_piece(piece, pre_processed_image, debug=args.debug)
            if not ok:
                print(f"Processing of piece {info.id} failed. Skipping this piece.")

            triangles = piece.rot['triangles']
            rotation_angle_rad = piece.rot['rotation_angle_rad']
            inverse_transform_fn = piece.orig['inverse_transform_fn']
            cx,cy = piece.orig['rot_center']
            corners = piece.rot['corners']
            tab_keep_outs = piece.rot['tab_keep_outs']
            blank_keep_outs = piece.rot['blank_keep_outs']
            corners = piece.rot['corners']

            # display triangles on orig image
            for p_tri in triangles:
                orig_tri_pts = [ rotate_and_transform_point(pts3, (cx,cy), -rotation_angle_rad, inverse_transform_fn) for pts3 in p_tri.points ]
                if len(p_tri.points) == 3:
                    draw_triangle(resized_image, orig_tri_pts, color=(255,255,0), thickness=2)
                else:
                    assert len(p_tri.points)==2
                    cv2.line(resized_image, (int(orig_tri_pts[0][0]), int(orig_tri_pts[0][1])), (int(orig_tri_pts[1][0]), int(orig_tri_pts[1][1])), color=(30,30,30), thickness=4)

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

            # Also add tabs and blank keep outs to original image
            for bbox in tab_keep_outs + blank_keep_outs:
                tl_bbox, br_bbox = bbox
                tr_bbox = (br_bbox[0], tl_bbox[1])
                bl_bbox = (tl_bbox[0], br_bbox[1])
                pts = [tl_bbox, br_bbox, tr_bbox, bl_bbox]
                orig_pts = [ rotate_and_transform_point(pt, (cx,cy), -rotation_angle_rad, inverse_transform_fn) for pt in pts ]
                draw_poly(resized_image, orig_pts, color=(250, 0, 0), thickness=4)
        
            # Put number on piece
            cv2.putText(resized_image, f"{idx}", (info.centroid[0] - 20, info.centroid[1] + 20) , cv2.FONT_HERSHEY_SIMPLEX, 3.0, (50, 50, 50), 5)

            # Show centroid
            cv2.circle(resized_image, info.centroid, 5, (200,30,30), 6)

        show_image(resized_image, "Orig with corners.", max_side=2000, wait_for_key=True)

        jigsaw = Jigsaw()
        jigsaw.unplaced = pieces

        size_r = re.split(r'x', args.size, maxsplit=1, flags=re.IGNORECASE)
        if size_r: # size_r[0] is width, size_r[1] is height
            jigsaw.set_dim(int(size_r[0]), int(size_r[1]))
        else:
            print("Error - jigsaw dimensions must be set using -s WxH.")
            exit(1)
        res = solve_puzzle(jigsaw)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Piece Project Initialized")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")

if __name__ == "__main__":
    main()
