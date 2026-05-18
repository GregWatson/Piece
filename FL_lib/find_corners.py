import numpy as np
from FL_lib.fl_core import get_angle_diff, get_distance_between_2_points

# Given a list of lines that form the edges of a piece, tryo to find
# the corners of a piece.
# This is a simplified approach that assumes the corners are where lines meet
# at a "significant" angle (e.g. 90 degrees).
# Params:
# - lines_found: list of lines, where each line is represented as a tuple of (points, angle)
# - corner_thresh: minimum angle difference in degrees to consider a corner (e.g. 50 degrees)
# - end_to_end_dist_thresh: maximum distance between the end of one line and the 
#   start of the next line to consider them connected (e.g. 20 pixels)
def find_corners(lines_found, corner_thresh=50, end_to_end_dist_thresh=20, debug=False):
    corners = []
    angle_thresh_rad = np.radians(corner_thresh)
    for i, line1 in enumerate(lines_found):
        line2 = lines_found[(i + 1) % len(lines_found)]  # wrap around to the first line     
        angle_diff = get_angle_diff(line1[1], line2[1])
        if angle_diff < angle_thresh_rad:
            if debug:
                print(f"Lines {i} and {(i+1)%len(lines_found)} are not a corner. Angle difference is {np.degrees(angle_diff):.2f} degrees.")
            continue  # not a corner
        line_separation = get_distance_between_2_points(line1[0][-1], line2[0][0])
        if line_separation > end_to_end_dist_thresh:
            if debug:
                print(f"Lines {i} and {(i+1)%len(lines_found)} are not a corner. End-to-end distance is {line_separation:.2f} pixels.")
            continue  # not a corner
        line1_length = get_distance_between_2_points(line1[0][0], line1[0][-1])
        line2_length = get_distance_between_2_points(line2[0][0], line2[0][-1])
        # compute a corner point as a function of the length of the lines and the angle between them.
        corner_x = line1_length * line2_length * angle_diff
        corners.append((corner_x, line1[0][-1], angle_diff))

    corners.sort(key=lambda x: x[0], reverse=True)  # sort by corner_x value, which is a function of line lengths and angle difference, so that the most "corner-like" corners are first.
    for i, corner in enumerate(corners):
        if debug:
            print(f"Corner {i}: corner_x={corner[0]:.2f}, point={corner[1]}, angle_diff={np.degrees(corner[2]):.2f} degrees")

    return corners

