"""Static city geometry and geodesic distances to interaction zones."""

from heapq import heappop, heappush
from math import hypot, sqrt

import numpy as np


WORK, HOME, JOBS, BANK, SLOTS, BLACKJACK, GARAGE = range(7)
ZONE_NAMES = ("Work", "Home", "Job centre", "Bank", "Slots", "Blackjack", "Garage")
WIDTH, HEIGHT = 30, 20
ZONES = ((10, 4), (15, 4), (24, 14), (5, 12), (10, 13), (18, 13), (25, 4))
START = (4, 4)
TRAFFIC = ((12, 8.5), (22, 8.5), (15, 17))
TRAFFIC_VELOCITY = ((1.2, 0), (-1.0, 0), (-0.9, 0))
TRAFFIC_LIMITS = ((8, 19), (19, 26), (9, 22))
WALLS = (
    (3, 8), (3, 9), (3, 10), (13, 10), (13, 11), (13, 12), (13, 13),
    (20, 6), (20, 7), (20, 8), (21, 12), (21, 13), (21, 14), (21, 15),
    (5, 16), (6, 16), (24, 10), (25, 10), (26, 10),
)
COLLISION_RADIUS = 1.0
BOUNDARY_MARGIN = 0.55
NAV_CLEARANCE = 1.12


def distance_fields(zone_radius):
    """Run Dijkstra once per zone on the static map in world coordinates."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    walls = np.asarray(WALLS)
    blocked = np.any(
        (xx[..., None] - walls[:, 0]) ** 2 + (yy[..., None] - walls[:, 1]) ** 2
        < NAV_CLEARANCE**2,
        axis=-1,
    )
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    moves = [(dx, dy, hypot(dx, dy)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    fields = []
    for zx, zy in ZONES:
        dist = np.full((HEIGHT, WIDTH), np.inf)
        starts = np.argwhere(((xx - zx)**2 + (yy - zy)**2 <= zone_radius**2) & ~blocked)
        queue = [(0., int(x), int(y)) for y, x in starts]
        dist[starts[:, 0], starts[:, 1]] = 0.
        while queue:
            cost, x, y = heappop(queue)
            if cost > dist[y, x]:
                continue
            for dx, dy, length in moves:
                nx, ny = x + dx, y + dy
                if blocked[ny, nx] or (dx and dy and (blocked[y, nx] or blocked[ny, x])):
                    continue
                candidate = cost + length
                if candidate < dist[ny, nx]:
                    dist[ny, nx] = candidate
                    heappush(queue, (candidate, nx, ny))
        if np.any(~np.isfinite(dist[~blocked])):
            raise ValueError("The city has a road disconnected from a service zone")
        # Bilinear interpolation near a wall needs finite values at blocked cells.
        for y, x in np.argwhere(blocked):
            neighbours = dist[max(0, y-1):y+2, max(0, x-1):x+2]
            nearest = neighbours[np.isfinite(neighbours)]
            if nearest.size:
                dist[y, x] = nearest.min() + sqrt(2.)
        fields.append(np.where(np.isfinite(dist), dist, WIDTH + HEIGHT))
    return np.asarray(fields, dtype=np.float32)
