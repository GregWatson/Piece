# Given a list of unordered lines, return the ordered list, where ordering
# tries to find the next closest line to the previous one.
import copy
import math
import numpy as np

def order_lines(unordered_lines):

    # NOTE: the starting line should probably be the longest line!

    print(f"Ordering list of {len(unordered_lines)} lines. {type(unordered_lines)}")
    rem_lines = [((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in unordered_lines]
    o_lines = [rem_lines.pop()]
    pt = o_lines[0][1]
    while len(rem_lines) > 1:
        # find a point in rem_lines closest to pt
        closest_dist = float('inf')
        for li,l in enumerate(rem_lines):
            for i,p in enumerate(l):
                d = math.sqrt((p[0]-pt[0])**2 + (p[1]-pt[1])**2)
                if d < closest_dist:
                    closest_dist = d
                    next_pt = l[1-i]  # next point is the other end of this line
                    next_line_index = li
        next_line = rem_lines[next_line_index]
        print(f"Closest line to {pt[0]},{pt[1]} is line {next_line[0][0]},{next_line[0][1]} - {next_line[1][0]},{next_line[1][1]} (dist={closest_dist}).  New pt is {next_pt[0]},{next_pt[1]}")
        pl = rem_lines.pop(next_line_index)
        o_lines.append(pl)
        pt = next_pt
    # last line is in rem_lines
    if len(rem_lines) > 0:
        o_lines.append(rem_lines[0])
        print(f"Last line is {rem_lines[0][0]} - {rem_lines[0][1]}")

    return o_lines
