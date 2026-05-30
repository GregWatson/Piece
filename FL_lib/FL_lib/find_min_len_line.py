import numpy as np
from FL_lib.fl_core import get_adjacent_points, get_angle, get_angle_diff, get_angle_tol

WHITE = 255
BLACK = 0

# Given a starting point, find a line that is at least len_thresh pixels in length.
# Algorithm is by following adjacent points in the grayscale image and checking how straight the line is. 
# We use a recursive backtracking approach to try different paths until we find a valid line or exhaust all possibilities.
# Stop once we have a line >= len_thresh.
# On success, returns the list of points along the line and the sum of X,Y coordinates of unit vectors of the angles of the points so far.
# - sum_unitXY_so_far is the sum of the X and Y of unit vectors of the angles of the points so far, 
#   which we use to calculate the average angle as we go along without having to recalculate it from scratch each time.
# On failure return ONLY the LAST point we looked at.
def find_min_len_line(start_point, points_so_far, gray, sum_unitXY_so_far = (0,0), len_thresh=10, debug=False):
    current_point = points_so_far[-1]
    gray[current_point[1], current_point[0]] = BLACK

    raw_adjacent_points = get_adjacent_points(gray, current_point)
    if len(raw_adjacent_points) == 0:
        return [current_point], None
    
    for pt in raw_adjacent_points:
        gray[pt[1], pt[0]] = BLACK

    # sort adjacent points by angle relative to angle_so_far
    avg_angle_so_far = np.arctan2(sum_unitXY_so_far[1]/len(points_so_far), sum_unitXY_so_far[0]/len(points_so_far)) if len(points_so_far) > 1 else 0
    adjacent_points = [(angle, pt) for pt in raw_adjacent_points for angle in [get_angle(pt, start_point)]]
    adjacent_points.sort(key=lambda x: get_angle_diff(x[0], avg_angle_so_far))

    # try the adjacent points until we find a valid line or we fail them all
    for angle, next_point in adjacent_points:
        if debug:
            print(f"Trying adjacent point {next_point} of {len(raw_adjacent_points)} from current point {current_point}. Angle is {np.degrees(angle):.1f} degrees, avg angle so far is {np.degrees(avg_angle_so_far):.1f} degrees.")

        if len(points_so_far) > 1: # check angle deviation from average so far
            angle_tol = get_angle_tol(len(points_so_far))  # tolerance decreases as line gets longer
            angle_diff = get_angle_diff(angle, avg_angle_so_far)
            if angle_diff > angle_tol:
                if debug:
                    print(f"  Angle difference {np.degrees(angle_diff):.1f} exceeds tolerance {np.degrees(angle_tol):.1f}. Skipping point {next_point}.")
                continue
            # Angle ok. Check length
            if len(points_so_far) >= len_thresh:
                new_sum_unitXY = (sum_unitXY_so_far[0] + np.cos(angle), sum_unitXY_so_far[1] + np.sin(angle))
                if debug:
                    new_avg_angle = np.arctan2(new_sum_unitXY[1], new_sum_unitXY[0])
                    print(f"  Found line of length {len(points_so_far)} with avg angle {np.degrees(new_avg_angle):.1f} deviation {np.degrees(angle_diff):.1f} within tolerance. Returning line.")
                # restore adjacent points that were marked as used but not part of the line
                for pt in raw_adjacent_points:
                    if pt != next_point:
                        gray[pt[1], pt[0]] = WHITE
                return points_so_far + [next_point], new_sum_unitXY

            # OK, find next point and keep looking for a line recursively
            if debug:
                print(f"  Angle difference {np.degrees(angle_diff):.1f} within tolerance {np.degrees(angle_tol):.1f}. Continuing to point {next_point}.")

        all_points, new_sum_unitXY = find_min_len_line(start_point, points_so_far + [next_point], gray, (sum_unitXY_so_far[0] + np.cos(angle), sum_unitXY_so_far[1] + np.sin(angle)), len_thresh, debug)
        if len(all_points) > 1: # we found a line
            for pt in raw_adjacent_points:
                if pt not in all_points:
                    gray[pt[1], pt[0]] = WHITE
            return all_points, new_sum_unitXY
        else:
            if debug:
                print(f"  Backtracking from point {next_point} to {current_point}.")
            # restore adjacent points that were marked as used but not part of the line

    # tried all adjacent points and couldn't find a valid line. Restore them and return failure.
    for pt in raw_adjacent_points:
        gray[pt[1], pt[0]] = WHITE
    return [current_point], None

