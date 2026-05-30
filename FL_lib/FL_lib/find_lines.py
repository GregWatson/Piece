import cv2
import numpy as np
import sys
from FL_lib.fl_core import find_initial_start_point, find_local_start_point, get_angle_diff, get_angle_tol, points_are_close, get_angle
from FL_lib.find_line import find_line

# Returns a list of lines found within the given image (BGR color or grayscale)
# where each line is represented as a tuple of (points, angle) 
# and points is a list of (x, y) coordinates along the line, 
# and angle is the average angle of the line in radians. 
# The len_thresh parameter specifies the minimum length of a line to be considered valid. 
# Lines shorter than this threshold will be discarded.
# The image should contain only edges or contours that form the boundaries of objects.

BLACK = 0

def find_lines(img, len_thresh=10, debug=False):
    lines_found = []
    # Convert the image to simple binary format (grayscale) if color
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    #find a starting point for the line
    start_point = find_initial_start_point(gray)
    if debug: print(f"Initial start point: {start_point}")

    while start_point:

        points, angle = find_line(start_point, gray, len_thresh=len_thresh, debug=debug)

        gray[start_point[1], start_point[0]] = BLACK  # remove the starting point from the image
        last_point = start_point # in case there was no line found.
        if len(points)>1: # if we found a valid line
            last_point = points[-1]
            merged = False
            # If this line is very similar in angle to the last line, and close to it, we can consider it part of the same line.
            if len(lines_found) > 0:
                prev_start_point = lines_found[-1][0][0]
                prev_last_point = lines_found[-1][0][-1]
                if points_are_close(points[0], prev_last_point,thresh=3):
                    angle_diff = get_angle_diff(angle, lines_found[-1][1])
                    if debug: print(f"   End Points are close. Angle difference between current line and last line: {np.degrees(angle_diff):.2f} degrees   ({np.degrees(angle):.2f} vs {np.degrees(lines_found[-1][1]):.2f} degrees)")
                    # angle_tol = min(get_angle_tol(len(lines_found[-1][0])), get_angle_tol(len(points)))
                    angle_tol = np.radians(6)  # we can use a fixed angle tolerance here since we are merging lines after they are fully formed, so we don't need to worry about the tolerance being too high at the start of a line.
                    if debug: print(f"   Angle tolerance for merging these lines is: {np.degrees(angle_tol):.2f} degrees. Actual diff is {np.degrees(angle_diff):.2f} degrees.")
                    if angle_diff < angle_tol:
                        # If the angle is within the tolerance, we can merge this line with the last line.
                        new_angle = get_angle(last_point, prev_start_point)
                        new_len = np.linalg.norm(np.array(last_point) - np.array(prev_start_point))
                        lines_found[-1] = (lines_found[-1][0] + points, new_angle)
                        merged = True
                        if debug: print("Merging current line with last line. New angle is {:.2f} degrees, new length is {:.2f}".format(np.degrees(new_angle), new_len))
                else:
                    if debug: print(f"   End points {prev_last_point} and {points[0]} are not close enough to merge the lines.")
            if merged == False:
                lines_found.append((points, angle))
                if debug: print(f"No merge: Adding new line with angle {np.degrees(angle):.2f} degrees and length {len(points)}")
        if len(points) == 1:
            last_point = points[0]

        # if debug:
        #     scaled_img = cv2.resize(gray, (500, 500), interpolation=cv2.INTER_NEAREST)
        #     cv2.imshow("Initial", scaled_img)
        #     cv2.waitKey(0)

        # find the next start point somewhere close to the last point.
        start_point = find_local_start_point(gray, last_point, size_thresh=50, color_thresh=127)
        if start_point: 
            if debug: print(f"Found next start point: {start_point} which was close to last point {last_point}.") 
        else:
            if debug:
                print(f"Couldn't find a new start point near last end point {int(last_point[0])},{int(last_point[1])} - searching entire image.")
                # img = gray.copy()
                # cv2.circle(img, (int(last_point[0]), int(last_point[1])), 5, (255, 255, 255), -1)
                # cv2.imshow(f"Image {int(last_point[0])},{int(last_point[1])}", img)
                # cv2.waitKey(0)
            start_point = find_initial_start_point(gray)
    return lines_found