import cv2
import numpy as np
from Process.order_lines import order_lines

# Find rotation by rotating the image from -45 to +45 degress and using the
# rotation with the minimum Bounding Box area.
# cx,cy is the center of mass of the piece which is one choice for center of rotation.
# Another choice is the center of the initial bounding box.
def find_rotation_by_BB(img, cx, cy):
    if img is None:
        print("Error: Image is None")
        return 0

    gray = img

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find a set of lines for the outline using Hough Transform (Probabilistic)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=20 , maxLineGap=10)
    o_lines = order_lines(lines)

    # Find corners from o_lines.


    line_count = 0
    min_angle = 0
    
    if lines is not None:
        line_count = len(lines)

        # Find the bounding box of the piece using line end points. This is a rough approximation but should be good enough for rotation estimation.
        bb_cx = 0
        bb_cy = 0
        bb_min_x = float('inf')
        bb_max_x = float('-inf')
        bb_min_y = float('inf') 
        bb_max_y = float('-inf')

        for line in lines:
            x1, y1, x2, y2 = line[0]
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


        # See if we have any LONG lines that are EDGES.
        len_threshold = (bb_xlength + bb_ylength) * 0.2
        line_max_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # print(f"Line segment is {x1},{y1} - {x2},{y2}")
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > len_threshold:
                if length > line_max_length:
                    line_max_length = length
                    # get angle
                    min_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    print(f"Longer line found: ({x1}, {y1}) to ({x2}, {y2}) with length {length}   (thresh: {len_threshold})  angle {min_angle}")

        # If we couldn't find a long line then we will just use the bounding box method.

        if line_max_length == 0:
            print("No long lines found. Using bounding box method to find rotation.")
            # translate the set of line endings so that cx,cy is at 0,0
            centered_points = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
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
                    min_angle = rot_angle
                    print(f"New min area {area} at angle {rot_angle}")

        # Create a copy for drawing
        display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mark the center of rotation
        cv2.circle(display_img, (cx, cy), 8, (0, 0, 255), -1) # Draw red circle
            
        cv2.imshow("Detected Lines", display_img)

    print(f"Lines found: {line_count}.   Angle {min_angle}")
    return min_angle, cx, cy, lines
