import numpy as np
# from Process.order_lines import order_lines
from FL_lib.find_lines import find_lines
from FL_lib.fl_core import get_angle, show_image
import cv2

# Find rotation by either:
# 1. Finding long lines that we can infer as edges, and using the angle of 
#    the longest line as the rotation. This is more accurate if we have a 
#    good long line, but may fail if the piece is very irregular or if the 
#    lines are not detected well.
#
# 2. Look at the straight lines and try to find a dominant angle by grouping 
#    lines into bins based on their angle and weighting them by their length. 
#    This is more robust to irregular pieces and line detection issues, but 
#    may be less accurate if there are a lot of short lines.
# 
# We can also use the center of mass of the piece as the center of rotation, 
# or we can use the center of the initial bounding box of the lines as the 
# center of rotation. The latter may be more stable if the piece is very 
# irregular and the center of mass is not a good representation of the piece's 
# actual center.
USE_CENTROID = True

def group_edge_lines(lines, line_lengths, num_bins=36, min_length=10):
    """
    Lines are list of (start, end) tuples from edge detection.
    line_lengths is length of corresponding line segments.
    Group lines into bins based on their angle, and weight them by their length.
    """
    if lines is None:
        return {}

    # Initialize empty bins: {bin_index: [list_of_lines]}
    angle_bins = { i: [] for i in range(num_bins)}
    bin_size_deg = 180.0 / num_bins

    for idx,line in enumerate(lines):
        (x1, y1), (x2, y2) = line
        dx = x2 - x1
        dy = y2 - y1
        length = line_lengths[idx]
        
        # Skip noise
        if length < min_length:
            continue
            
        # Get angle in degrees (0 to 180)
        angle_deg = np.degrees(np.arctan2(dy, dx)) % 180
        
        # Determine which bin it falls into
        bin_idx = int(angle_deg // bin_size_deg) % num_bins
        
        # Store the line along with its weight (length)
        angle_bins[bin_idx].append({
            'weight': length,
            'angle': angle_deg
        })
        
    return angle_bins

def get_sorted_bins(angle_bins):
    bin_weights = {}
    for bin_idx, lines in angle_bins.items():
        # Sum up the lengths of all lines in this specific bin
        total_weight = sum(line['weight'] for line in lines)
        bin_weights[bin_idx] = total_weight
    # Sort bins by total line length (highest weight first)
    sorted_bins = sorted(bin_weights.items(), key=lambda x: x[1], reverse=True)
    return sorted_bins

# Combine num_adj bins on either side of the max bin to get a more stable angle estimate, 
# and take the weighted average of the angles in those bins.
def get_main_angle(angle_bins,max1, num_adj=2, debug=False):
    if max1 not in angle_bins:
        return None
    main_bins = [max1]
    for i in range(1, num_adj+1):
        main_bins.append((max1 + i) % len(angle_bins))
        main_bins.append((max1 - i) % len(angle_bins))
    total_weight = 0
    weighted_angle_sum = 0
    for bin_idx in main_bins:
        for line in angle_bins[bin_idx]:
            weight = line['weight']
            angle_rad = np.radians(line['angle'])
            weighted_angle_sum += weight * angle_rad
            total_weight += weight
    if total_weight == 0:
        return None
    main_angle_rad = weighted_angle_sum / total_weight
    if debug:
        print(f"Main angle (degrees): {np.degrees(main_angle_rad)}")
    return main_angle_rad

# Input is the image and the center of mass of the piece.
# Returns the rotation angle in radians, and the center of rotation (cx, cy)
# We can also return the lines we found for debugging purposes - lines are 
# returned as a list of tuples of [(x1, y1), (x2, y2)] for the start and end points 
# of each line.

def find_rotation(img, cx, cy, debug=False):
    if img is None:
        print("Error: Image is None")
        return 0

    # Convert the image to simple binary format (grayscale) if color
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()


    # Find a set of lines for the outline using my own algorithm.
    full_lines = find_lines(gray, debug=debug)
    lines = []
    
    # Get lines as just start and end points for easier processing.
    if len(full_lines): 
        lines = [[(line[0][0][0], line[0][0][1]), (line[0][-1][0], line[0][-1][1])] for line in full_lines]

    line_count = len(full_lines)
    min_angle = 0
    
    if len(lines):
        line_count = len(lines)

        # Find the bounding box of the piece using line end points. This is a rough approximation but should be good enough for rotation estimation.
        bb_cx = 0
        bb_cy = 0
        bb_min_x = float('inf')
        bb_max_x = float('-inf')
        bb_min_y = float('inf') 
        bb_max_y = float('-inf')

        for line in lines:
            (x1, y1), (x2, y2) = line
            bb_cx += (x1 + x2) / 2
            bb_cy += (y1 + y2) / 2
            bb_min_x = min(bb_min_x, x1, x2)
            bb_max_x = max(bb_max_x, x1, x2)
            bb_min_y = min(bb_min_y, y1, y2)
            bb_max_y = max(bb_max_y, y1, y2)

        bb_cx = int(bb_cx / line_count)
        bb_cy = int(bb_cy / line_count)
        bb_xlength = bb_max_x - bb_min_x
        bb_ylength = bb_max_y - bb_min_y

        if debug:
            # print(f"center of mass x,y is {cx},{cy}")
            # print(f"center of bounding box x,y is {bb_cx},{bb_cy}.  bb X len={bb_xlength}  bb Y len={bb_ylength}")
            if USE_CENTROID:
                print(f"Using center of mass for rotation: {cx},{cy}")
            else:
                print(f"Using center of bounding box for rotation: {bb_cx},{bb_cy}")
        if not USE_CENTROID:
            cx = bb_cx
            cy = bb_cy

        # See if we have any LONG lines that we can infer as EDGES.
        len_threshold = (bb_xlength + bb_ylength) * 0.2
        line_max_length = 0
        min_angle = 0
        max_line = None
        line_lengths = []
        for line in lines:
            (x1, y1), (x2, y2) = line
            # print(f"Line segment is {x1},{y1} - {x2},{y2}")
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            line_lengths.append(length)
            if length > len_threshold:
                if length > line_max_length:
                    line_max_length = length
                    max_line = line
                    # get angle
            if line_max_length:
                min_angle = get_angle(max_line[1],max_line[0])
                if debug:
                    print(f"Longest line found: ({x1}, {y1}) to ({x2}, {y2}) with length {length}   (thresh: {len_threshold})  angle {min_angle}")

        # If we couldn't find a long line then we will just use the bounding box method.

        if line_max_length == 0:

            # min_angle,cx,cy = get_rot_by_longest_line(gray, cx, cy, debug=debug)
            angle_bins = group_edge_lines(lines, line_lengths, num_bins=12, min_length=10)
            sorted_bins = get_sorted_bins(angle_bins)
            min_angle = get_main_angle(angle_bins,sorted_bins[0][0], num_adj=2, debug=debug)

            if min_angle == None:
                if debug:
                    print("No long lines found. Using bounding box method to find rotation.")
                # translate the set of line endings so that cx,cy is at 0,0
                centered_points = []
                for line in lines:
                    (x1, y1), (x2, y2) = line
                    centered_points.append((x1 - cx, y1 - cy)) 
                    centered_points.append((x2 - cx, y2 - cy))

                min_angle = 0
                min_area = float('inf')

                for rot_angle in range(1, 45):
                    if rot_angle == 0: next
                    bbmin=(cx,cy)
                    bbmax=(cx,cy)
                    rot_rad = np.radians(rot_angle)
                    rot_matrix = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], 
                                        [np.sin(rot_rad), np.cos(rot_rad)]]) 
                    for point in centered_points:
                        rot_point = np.dot(rot_matrix, point)
                        bbmin = (min(bbmin[0], rot_point[0]), min(bbmin[1], rot_point[1]))
                        bbmax = (max(bbmax[0], rot_point[0]), max(bbmax[1], rot_point[1]))
                    area = (bbmax[0] - bbmin[0]) * (bbmax[1] - bbmin[1])
                    if area < min_area:
                        min_area = area
                        min_angle = rot_rad
                        # print(f"New min area {area} at angle {rot_angle} degrees")
    
    # Don't need a rotation > 90 degrees
    while np.degrees(min_angle) > 90:
        min_angle -= np.pi
        
    if debug:
        print(f"Lines found: {line_count}.   Angle {np.degrees(min_angle):.2f} degrees")
    return min_angle, cx, cy, lines
