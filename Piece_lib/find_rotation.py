import numpy as np
# from Process.order_lines import order_lines
from FL_lib.find_lines import find_lines
from FL_lib.fl_core import get_angle

# Find rotation by either:
# 1. Finding long lines that we can infer as edges, and using the angle of 
#    the longest line as the rotation. This is more accurate if we have a 
#    good long line, but may fail if the piece is very irregular or if the 
#    lines are not detected well.
# 2. If we can't find a long line, we can rotate the image from -45 to +45 
#    degrees and find the bounding box of the lines at each angle, and use 
#    the angle that gives us the minimum bounding box area as the rotation. 
#    This is less accurate but more robust to irregular pieces and line 
#    detection issues
# 
# We can also use the center of mass of the piece as the center of rotation, 
# or we can use the center of the initial bounding box of the lines as the 
# center of rotation. The latter may be more stable if the piece is very 
# irregular and the center of mass is not a good representation of the piece's 
# actual center.

# Input is center of mass of piece, and the image. 
# Returns the rotation angle in radians, and the center of rotation (cx, cy)
# We can also return the lines we found for debugging purposes - lines are 
# returned as a list of tuples of [(x1, y1), (x2, y2)] for the start and end points 
# of each line.

def find_rotation(img, cx, cy):
    if img is None:
        print("Error: Image is None")
        return 0

    gray = img

    # Find a set of lines for the outline my own algorithm.
    full_lines = find_lines(gray)
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

        print(f"center of mass x,y is {cx},{cy}")
        print(f"center of bounding box x,y is {bb_cx},{bb_cy}.  bb X len={bb_xlength}  bb Y len={bb_ylength}")
        print(f"Using center of bounding box for rotation: {bb_cx},{bb_cy}")
        cx = bb_cx
        cy = bb_cy

        # See if we have any LONG lines that we can infer as EDGES.
        len_threshold = (bb_xlength + bb_ylength) * 0.2
        line_max_length = 0
        min_angle = 0
        max_line = None
        for line in lines:
            (x1, y1), (x2, y2) = line
            # print(f"Line segment is {x1},{y1} - {x2},{y2}")
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > len_threshold:
                if length > line_max_length:
                    line_max_length = length
                    max_line = line
                    # get angle
            if line_max_length:
                min_angle = get_angle(max_line[3] - max_line[1], max_line[2] - max_line[0])
                print(f"Longest line found: ({x1}, {y1}) to ({x2}, {y2}) with length {length}   (thresh: {len_threshold})  angle {min_angle}")

        # If we couldn't find a long line then we will just use the bounding box method.

        if line_max_length == 0:
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
                    print(f"New min area {area} at angle {rot_angle} degrees")

    print(f"Lines found: {line_count}.   Angle {np.degrees(min_angle):.2f} degrees")
    return min_angle, cx, cy, lines
