import numpy as np
from FL_lib.fl_core import get_angle, get_angle_diff, get_distance_between_2_points
from FL_lib.polygon_angles import get_polygon_angles
import cv2

# Given a list of lines that form the edges of a piece, try to find
# the corners of a piece.
# This is a simplified approach that assumes the corners are where lines meet
# at a "significant" angle (e.g. 90 degrees).
# Params:
# - lines_found: list of lines, where each line is represented as a tuple of start and end points.
#   The lines form the outline of the piece in clockwise order. The end point of a line might not 
#   be the same as the start point of the next line. If they are close then we treat them as the same, 
#   but if they are more than a threshold distance apart then we ignore the gap and don't consider
#   them as connected (cannot be a corner).
# - corner_thresh: minimum angle difference in degrees to consider a corner (e.g. 50 degrees)
# - end_to_end_dist_thresh: maximum distance between the end of one line and the 
#   start of the next line to consider them connected (e.g. 20 pixels)
# Returns a list of corners, where each corner is represented as a tuple of (corner_x, point, angle_diff) where:
# - corner_x is a value that represents how "corner-like" this corner is, which is a function of the 
#   angle difference and the lengths of the lines. We can use this to sort the corners and only keep the most "corner-like" corners.
# - point is the (int(x), int(y)) point of the corner (e.g. the end of line1 or the start of line2, or we could compute an intersection point 
#   if we want to be more precise)
# - angle_diff is the EXTERNAL angle in radians between the two lines that form this corner, which can be useful for 
#   debugging and for further filtering of corners if needed.

# Compute a corner-ish function to evaluate how "corner-like" each potential corner is
# This function will be used to sort the corners and select the most "corner-like" ones.
# The angle is the EXTERNAL angle (of a polygon) between the two lines that form the corner.
def cornerness_function(pt0, pt1, pt2, angle, rel_d_to_c, debug=False):
    l1 = get_distance_between_2_points(pt0, pt1)
    l2 = get_distance_between_2_points(pt1, pt2)
    if abs(angle) < np.pi: # if the angle is very small, it's not a corner
        corner_ness = 0
    else:
        # angle_effect = 10.0/(abs(np.pi*1.5 - abs(angle))) # the less the angle deviates from 270 degrees, the more corner-like it is
        angle_effect = 20-20**(abs(np.pi*1.5 - abs(angle))) # the less the angle deviates from 270 degrees, the more corner-like it is
        length_effect = (l1 + l2)/2 # longer lines should contribute to more corner-ness, but we can take the average to avoid giving too much weight to one long line and one short line
        dist_to_corner_effect = 5.0 * (30 - 30**rel_d_to_c) # the closer to the corner of the image, the more likely it is to be a real corner of the piece rather than noise in the middle
        corner_ness = angle_effect + length_effect + dist_to_corner_effect
        if debug:
            print(f"At {int(pt1[0])},{int(pt1[1])}: angle={int(np.degrees(angle))}, lens={int(l1)}, {int(l2)}. d2c= {rel_d_to_c:.2f} ", end=' ')
            print(f"angle_effect = {angle_effect:.3f}, length_effect = {length_effect:.3f}, dist_to_corner_effect = {dist_to_corner_effect:.3f}", end=' ')
            print(f" ===> Corner_ness = {corner_ness:.2f}", end=' ')
    return corner_ness

def get_rel_shortest_distance_to_corner (pt, w,h, half_diag):
    corners = [(0,0), (w,0), (w,h), (0,h)]
    m = min([get_distance_between_2_points(pt, corner) for corner in corners])
    return m/half_diag

def find_corners(lines_found, tl_bbox, br_bbox, end_to_end_dist_thresh=20, debug=False):
    corners = []
    if not len(lines_found):
        return corners

    # First, create a fully closed polygon from the lines. If the end of one line is close to the start of 
    # the next line, we consider them connected and we change the points to be the average of the two points.
    # If they are too far apart then we introduce a fake line to just close the polygon but will 
    # not be used in the corner detection.

    # if debug:
    #     for i,l in enumerate(lines_found):
    #         print(f"Line: {int(l[0][0])},{int(l[0][1])} ==> {int(l[1][0])},{int(l[1][1])}")

    polygon = []
    for i in range(len(lines_found)):
        line1 = lines_found[i]
        line2 = lines_found[(i + 1) % len(lines_found)]
        # if debug:
        #    print(f"Processing vertex {i}: ",end='')
        end1 = line1[1]
        start2 = line2[0]
        distance = get_distance_between_2_points(end1, start2)
        # if debug:
        #     print(f" Distance {int(end1[0])}, {int(end1[1])} - {int(start2[0])}, {int(start2[1])    }: {int(distance)}", end='')
        if distance <= end_to_end_dist_thresh:
            # If the end of the first line is close to the start of the second line, average the points
            avg_x = int((end1[0] + start2[0]) / 2)
            avg_y = int((end1[1] + start2[1]) / 2)
            polygon.append([avg_x, avg_y])
            # if debug:
            #     print(" So averaged to [" + str(avg_x) + ", " + str(avg_y) + "]")
        else:
            # If they are too far apart, we'll handle this later when checking for corners
            polygon.append([int(end1[0]), int(end1[1])])
            polygon.append([int(start2[0]), int(start2[1])])
            # if debug:
            #     print(f" So added as separate points. {polygon[-2:-1]}")

    # debug by displaying an image with points joined by arrows in order
    if debug and len(polygon) > 1:
        SIZE = 500
        img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)
        for i, point in enumerate(polygon):
            cv2.arrowedLine(img, (point[0], point[1]), (polygon[(i + 1) % len(polygon)][0], polygon[(i + 1) % len(polygon)][1]), (0, 150, 0), 1)
            cv2.putText(img, f"{i}", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow(f"Polygon ordering", img)
        # cv2.waitKey(0)

    poly_points = np.array(polygon, dtype=np.int32)
    angles = get_polygon_angles(poly_points)

    if debug:
        for i, angle in enumerate(angles):
            print(f"Vertex {i}: point={int(poly_points[i][0])},{int(poly_points[i][1])}, inner angle={int(np.degrees(angle[0]))}, outer angle={int(np.degrees(angle[1]))}")
    
    bb_w = br_bbox[0] - tl_bbox[0]
    bb_h = br_bbox[1] - tl_bbox[1]
    half_diag = np.sqrt(bb_w**2 + bb_h**2) / 2

    for i, angle in enumerate(angles):
        pt0 = poly_points[(i - 1) % len(poly_points)]
        pt1 = poly_points[i]
        pt2 = poly_points[(i + 1) % len(poly_points)]
        if debug: print(f"Corner {i}:", end='')
        rel_d_to_c = get_rel_shortest_distance_to_corner((pt1[0]-tl_bbox[0], pt1[1]-tl_bbox[1]), bb_w, bb_h, half_diag) 
        #print(f"Distance to closest image corner: {int(d_to_c)}")
        cornerness_value = cornerness_function(pt0, pt1, pt2, angle[1], rel_d_to_c, debug=debug) # outer angle
        # Store the corner information
        corners.append((cornerness_value, pt1, angle[1]))
        if debug:
            print("")
    # Sort the corners by their "corner-ness" (cornerness_value) in descending order
    corners.sort(key=lambda x: x[0], reverse=True)

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

    if debug:
        for i, corner in enumerate(final_corners):
            print(f"Corner {i}: corner_x={corner[0]:.2f}, point={corner[1]}, angle_diff={np.degrees(corner[2]):.2f} degrees")
        for i, corner in enumerate(final_corners):
            print(f"({corner[1][0]}, {corner[1][1]}), ", end='')
        print

    return final_corners

