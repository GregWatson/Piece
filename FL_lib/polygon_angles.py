import math
import numpy as np

def edge(a, b):
    """Return vector from point a to point b."""
    return (b[0] - a[0], b[1] - a[1])

def dot(u, v):
    """Return the dot product of vectors u and v."""
    return u[0]*v[0] + u[1]*v[1]

def cross(u, v):
    """Return the cross product of vectors u and v."""
    return u[0]*v[1] - u[1]*v[0]

def length(u):
    """Return the length of vector u."""
    return math.hypot(u[0], u[1])

def angle_between(u, v):
    """Return the unsigned angle between vectors u and v in degrees."""
    cos_theta = dot(u, v) / (length(u) * length(v))
    cos_theta = max(-1.0, min(1.0, cos_theta)) # clamp for safety
    return math.degrees(math.acos(cos_theta))

# Given a list of points determine if the ordering is counter-clockwise or clockwise. 
# This can be useful to determine if the angles we compute are inner or outer angles of a polygon.
def is_counter_clockwise(points):
    total = 0
    n = len(points)

    for i in range(n):
        # Wrap around to the first point for the last edge
        j = (i + 1) % n

        x1, y1 = points[i]
        x2, y2 = points[j]

        total += (x2 - x1) * (y2 + y1)

    # Positive means CCW, Negative means CW
    return total >= 0

    # Example usage:
    # points = [(0, 0), (2, 0), (2, 2), (0, 2)]
    # print(is_counter_clockwise(points)) # Returns True (Counter-Clockwise)


def get_polygon_angles(vertices):
    """
    Given a list of vertices in CW order, return a list of (inner, outer) angles.
    """
    n = len(vertices)
    results = []

    is_CCW = is_counter_clockwise(vertices)
    # print(f"Polygon is {'CCW' if is_CCW else 'CW'}")

    for i in range(n):
        p_prev = vertices[i - 1]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]

        u = edge(p_curr, p_prev) # incoming edge
        v = edge(p_curr, p_next) # outgoing edge

        theta = np.radians(angle_between(u, v))
        c = cross(u, v)

        if is_CCW:
            if c < 0:
                outer = theta
                inner = np.pi * 2 - theta
            else:
                outer = np.pi * 2 - theta
                inner = theta
        else:
            if c >= 0:
                outer = theta
                inner = np.pi * 2 - theta
            else:   
                outer = np.pi * 2 - theta
                inner = theta

        # print(f"ANGLE: Vertex {i}: point={int(p_curr[0])},{int(p_curr[1])}, outer angle={int(np.degrees(outer))}, inner angle={int(np.degrees(inner))}, cross={int(c)}")
        results.append((inner, outer))

    return results