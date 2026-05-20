# core routines used for line finding
import numpy as np

def rotate_point(point, center, angle_rad):
    # Rotate a point around a center by a given angle in radians.
    # Returns the new coordinates of the point after rotation.
    x, y = point
    cx, cy = center
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    new_x = cos_a * (x - cx) - sin_a * (y - cy) + cx
    new_y = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return (new_x, new_y)

def rotate_line(line, center, angle_rad):
    # Rotate a line (defined by two points) around a center by a given angle in radians.
    # Returns the new coordinates of the line after rotation.
    (x1, y1), (x2, y2) = line
    new_start = rotate_point((x1, y1), center, angle_rad)
    new_end = rotate_point((x2, y2), center, angle_rad)
    return (new_start, new_end)

def get_distance_between_2_points(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def points_are_close(pt1, pt2, thresh=10):
    return get_distance_between_2_points(pt1, pt2) < thresh

# compute angular tolerance based on line length, and use it to check if the angle of a new point 
# is consistent with the average angle of the line so far.
def get_angle_tol(line_length):
    return abs(np.arctan2(1.8, line_length)) if line_length > 0 else np.pi/2
    

# Calculate angle in radians between the line from start_point to pt and the horizontal axis.
# Always returns a positive angle between 0 and 2*pi.
def get_angle(pt, start_point):
    a = np.arctan2(pt[1] - start_point[1], pt[0] - start_point[0])
    while a < 0.0:
        a += 2*np.pi
    return a

# Calculate the smallest difference between two angles in radians, accounting for wraparound at 2*pi.
def get_angle_diff(angle1, angle2):
    diff = abs(angle1 - angle2)
    if diff > np.pi:
        diff = abs(2*np.pi - diff)
    return diff

def find_initial_start_point(gray):
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if gray[y, x] >= 127:
                return (x, y)
    return None

# Search an ever-expanding circle around the last point, looking for a set pixel.
def find_local_start_point(gray, last_point, size_thresh=10, color_thresh=255):
    # print(f"Searching for local start point around {last_point} with size_thresh {size_thresh} and color_thresh {color_thresh}.")
    x, y = last_point
    for dist in range(1, size_thresh + 1):
        for new_y in range(y - dist, y + dist + 1):
            if new_y in [y - dist, y + dist]:
                for new_x in range(x-dist, x+dist+1):
                    if 0 <= new_x < gray.shape[1] and 0 <= new_y < gray.shape[0]:
                        if gray[new_y, new_x] >= color_thresh:
                            return (new_x, new_y)
            else:
                for new_x in [x - dist, x + dist]:
                    if 0 <= new_x < gray.shape[1] and 0 <= new_y < gray.shape[0]:
                        if gray[new_y, new_x] >= color_thresh:
                            return (new_x, new_y)

    return None


def get_adjacent_points(gray, point, thresh=255):
    x, y = point
    adjacent_points = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < gray.shape[1] and 0 <= new_y < gray.shape[0]:
                if gray[new_y, new_x] >= thresh:
                    adjacent_points.append((new_x, new_y))
    return adjacent_points

# returns 6 color palette and names. Colors are (B:G:R) tuples.
def get_palette(palette_size=6):
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    names = ["Blue", "Green", "Red", "Cyan", "Magenta", "Yellow"]
    return (palette[:palette_size], names[:palette_size])