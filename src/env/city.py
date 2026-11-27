"""Fixed service locations and obstacle-aware navigation fields."""

from heapq import heappop, heappush
from math import hypot, sqrt

import numpy as np


HOME, WORK, BANK, GARAGE, SLOTS, BLACKJACK, FUEL, DINER, CAFE, RESTAURANT = range(10)
NAMES = ("Home", "Work", "Bank", "Garage", "Slots", "Blackjack",
         "Fuel station", "Budget diner", "Cafe", "Restaurant")
WIDTH, HEIGHT = 40, 30
ZONES = ((5, 15), (13, 15), (18, 25), (25, 26), (20, 6), (24, 6),
         (6, 8), (10, 8), (34, 11), (34, 19))
START = ZONES[HOME]
WALLS = tuple((28, y) for y in (*range(3, 12), *range(19, 27))) + (
    (16, 5), (16, 6), (16, 7), (16, 8), (16, 9), (16, 10),
    (21, 19), (22, 19), (23, 19), (11, 23), (11, 24), (11, 25),
)
TRAFFIC = ((15, 12), (22, 22), (31, 15))
TRAFFIC_VELOCITY = ((.65, 0), (-.55, 0), (.45, 0))
TRAFFIC_LIMITS = ((8, 24), (15, 26), (30, 37))
# The assignment models both vehicle and obstacles as circles of radius one.
COLLISION_RADIUS = 2.0
BOUNDARY_MARGIN = 1.05


def distance_fields(radius):
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    walls = np.asarray(WALLS)
    blocked = np.any((xx[..., None]-walls[:, 0])**2 +
                     (yy[..., None]-walls[:, 1])**2 < (COLLISION_RADIUS+.12)**2, axis=-1)
    blocked[[0, 1, -1, -2], :] = True
    blocked[:, [0, 1, -1, -2]] = True
    fields = []
    for zx, zy in ZONES:
        distance = np.full((HEIGHT, WIDTH), np.inf)
        starts = np.argwhere(((xx-zx)**2+(yy-zy)**2 <= radius**2) & ~blocked)
        queue = [(0., int(x), int(y)) for y, x in starts]
        distance[starts[:, 0], starts[:, 1]] = 0.
        while queue:
            cost, x, y = heappop(queue)
            if cost > distance[y, x]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x+dx, y+dy
                if blocked[ny, nx] or (dx and dy and (blocked[y, nx] or blocked[ny, x])):
                    continue
                candidate = cost+hypot(dx, dy)
                if candidate < distance[ny, nx]:
                    distance[ny, nx] = candidate
                    heappush(queue, (candidate, nx, ny))
        if np.any(~np.isfinite(distance[~blocked])):
            raise ValueError("A service is disconnected from the road network")
        for y, x in np.argwhere(blocked):
            neighbours = distance[max(0, y-1):y+2, max(0, x-1):x+2]
            finite = neighbours[np.isfinite(neighbours)]
            if finite.size:
                distance[y, x] = finite.min()+sqrt(2.)
        fields.append(np.where(np.isfinite(distance), distance, WIDTH+HEIGHT))
    return np.asarray(fields, dtype=np.float32)
