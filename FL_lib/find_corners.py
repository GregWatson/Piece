import numpy as np
from FL_lib.fl_core import get_angle, get_angle_diff, get_distance_between_2_points

# Given a list of lines that form the edges of a piece, try to find
# the corners of a piece.
# This is a simplified approach that assumes the corners are where lines meet
# at a "significant" angle (e.g. 90 degrees).
# Params:
# - lines_found: list of lines, where each line is represented as a tuple of start and end points
# - corner_thresh: minimum angle difference in degrees to consider a corner (e.g. 50 degrees)
# - end_to_end_dist_thresh: maximum distance between the end of one line and the 
#   start of the next line to consider them connected (e.g. 20 pixels)
# Returns a list of corners, where each corner is represented as a tuple of (corner_x, point, angle_diff) where:
# - corner_x is a value that represents how "corner-like" this corner is, which is a function of the 
#   angle difference and the lengths of the lines. We can use this to sort the corners and only keep the most "corner-like" corners.
# - point is the point of the corner (e.g. the end of line1 or the start of line2, or we could compute an intersection point 
#   if we want to be more precise)
# - angle_diff is the angle difference in radians between the two lines that form this corner, which can be useful for 
#   debugging and for further filtering of corners if needed.

def find_corners(lines_found, corner_thresh=50, end_to_end_dist_thresh=20, debug=False):
    corners = []
    angle_thresh_rad = np.radians(corner_thresh)
    for i, line1 in enumerate(lines_found):
        line2 = lines_found[(i + 1) % len(lines_found)]  # wrap around to the first line     
        angle1 = get_angle(line1[1], line1[0])  # angle of line1 from start to end
        angle2 = get_angle(line2[1], line2[0])  # angle of line2 from start to end (reversed since we want the angle at the corner)
        angle_diff = get_angle_diff(angle1, angle2)
        if angle_diff < angle_thresh_rad:
            if debug:
                print(f"Lines {i} and {(i+1)%len(lines_found)} are not a corner. Angle difference is {np.degrees(angle_diff):.2f} degrees.")
            continue  # not a corner
        line_separation = get_distance_between_2_points(line1[1], line2[0])
        if line_separation > end_to_end_dist_thresh:
            if debug:
                print(f"Lines {i} and {(i+1)%len(lines_found)} are not a corner. End-to-end distance is {line_separation:.2f} pixels.")
            continue  # not a corner
        line1_length = get_distance_between_2_points(line1[0], line1[1])
        line2_length = get_distance_between_2_points(line2[0], line2[1])
        # compute a corner point as a function of the length of the lines and the angle between them.
        corner_x = line1_length * line2_length * angle_diff
        corners.append((corner_x, line1[1], angle_diff))

    corners.sort(key=lambda x: x[0], reverse=True)  # sort by corner_x value, which is a function of line lengths and angle difference, so that the most "corner-like" corners are first.

    # Only have one corner per quadrant, and only keep corners that are above a certain threshold of "corner-ness" (e.g. corner_x > 0.5)
    final_corners = []  
    # compute the quadrant of each corner relative to the center of the piece, and only keep the most "corner-like" corner in each quadrant.
    quandrant_used= [[False, False],[False,False]]
    # get center
    center_x = np.mean([point[0] for _, point, _ in corners])
    center_y = np.mean([point[1] for _, point, _ in corners])
    for corner in corners:
        _,point,_ = corner
        # print(f"Evaluating corner with point={point}")
        quad_x = 0 if point[0] < center_x else 1
        quad_y = 0 if point[1] < center_y else 1
        if not quandrant_used[quad_y][quad_x] :
            final_corners.append(corner)
            quandrant_used[quad_y][quad_x] = True
            if debug:
                print(f"Adding corner at point {point} with corner_x value {corner[0]:.2f} and angle difference {np.degrees(corner[2]):.2f} degrees to final corners.")
        else:
            if debug:
                print(f"Skipping corner at point {point} as quadrant {quad_x},{quad_y} is used.")

    for i, corner in enumerate(final_corners):
        if debug:
            print(f"Corner {i}: corner_x={corner[0]:.2f}, point={corner[1]}, angle_diff={np.degrees(corner[2]):.2f} degrees")

    return final_corners

