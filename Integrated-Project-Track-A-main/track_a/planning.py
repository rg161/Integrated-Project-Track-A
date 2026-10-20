"""LiDAR-built occupancy map and A* waypoint path (no obstacle ground truth)."""

import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt
from .frames import rot_z


class LidarPlanner:
    def __init__(self, config):
        self.config = config
        self.res = config.grid_resolution
        x0, x1, y0, y1 = config.bounds
        self.origin = np.array([x0, y0])
        self.shape = (
            int(np.ceil((y1 - y0) / self.res)),
            int(np.ceil((x1 - x0) / self.res)),
        )
        self.log_odds = np.zeros(self.shape)
        self.blocked = np.zeros(self.shape, dtype=bool)
        self.clearance = np.full(self.shape, np.inf)
        self.path = np.empty((0, 2))
        self.angles = np.linspace(-np.pi, np.pi, config.lidar_beams, endpoint=False)

    def _cell(self, xy):
        c = np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)
        return int(c[1]), int(c[0])

    def _point(self, cell):
        return self.origin + (np.array([cell[1], cell[0]]) + 0.5) * self.res

    def _inside(self, cell):
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def update(self, pose, ranges):
        if ranges is None:
            return
        R = rot_z(pose[2])[:2, :2]
        origin = np.asarray(pose[:2]) + R @ np.array(self.config.lidar_pos[:2])
        for a, r in zip(self.angles, ranges):
            if np.isnan(r) or r <= 0:
                continue
            hit = np.isfinite(r) and r < self.config.lidar_max
            dist = min(r, self.config.lidar_max)
            end = origin + dist * np.array([np.cos(pose[2] + a), np.sin(pose[2] + a)])
            n = max(2, int(dist / self.res * 1.5))
            cells = np.floor(
                (np.linspace(origin, end, n) - self.origin) / self.res
            ).astype(int)
            good = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < self.shape[1])
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < self.shape[0])
            )
            free = np.unique(cells[:-2][good[:-2]], axis=0)
            self.log_odds[free[:, 1], free[:, 0]] -= 0.25
            c = self._cell(end)
            if hit and self._inside(c):
                self.log_odds[c] += 1.2
        np.clip(self.log_odds, -4, 4, out=self.log_odds)
        occupied = self.log_odds > 0.65
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
        distance = distance_transform_edt(~occupied) * self.res
        self.clearance = distance
        self.blocked = distance < self.config.robot_radius + self.config.safety_margin

    def _line_clear(self, a, b):
        n = max(2, int(np.linalg.norm(np.asarray(b) - a) * 2) + 1)
        cells = np.rint(np.linspace(a, b, n)).astype(int)
        return all(self._inside(tuple(c)) and not self.blocked[tuple(c)] for c in cells)

    def plan(self, start_xy, goal_xy):
        start, goal = self._cell(start_xy), self._cell(goal_xy)
        if not self._inside(start) or not self._inside(goal) or self.blocked[goal]:
            self.path = np.empty((0, 2))
            return self.path
        # Allow the robot's present grid cell as the starting node only.
        # If discretisation or pose correction puts the start inside the extra
        # planning margin, permit a monotonic escape through that margin. The
        # physical footprint remains forbidden. Otherwise an inflated start
        # cell can deadlock even though the robot has free space behind it.
        minimum_clearance = min(
            self.clearance[start], self.config.robot_radius + self.config.safety_margin
        )

        def traversable(c):
            return self._inside(c) and (
                not self.blocked[c]
                or (
                    self.clearance[c] >= minimum_clearance - 1e-9
                    and self.clearance[c] >= self.config.robot_radius
                )
            )

        frontier = [(0.0, start)]
        cost = {start: 0.0}
        parent = {}
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        found = False
        while frontier:
            _, cell = heapq.heappop(frontier)
            if cell == goal:
                found = True
                break
            for dy, dx in directions:
                nxt = (cell[0] + dy, cell[1] + dx)
                if not traversable(nxt):
                    continue
                if (
                    self.blocked[cell]
                    and self.clearance[nxt] < self.clearance[cell] - 1e-9
                ):
                    continue
                if (
                    dx
                    and dy
                    and (
                        not traversable((cell[0] + dy, cell[1]))
                        or not traversable((cell[0], cell[1] + dx))
                    )
                ):
                    continue  # do not cut an occupied grid corner
                newcost = cost[cell] + np.hypot(dx, dy)
                if newcost < cost.get(nxt, float("inf")):
                    cost[nxt] = newcost
                    parent[nxt] = cell
                    heapq.heappush(
                        frontier,
                        (newcost + np.hypot(nxt[0] - goal[0], nxt[1] - goal[1]), nxt),
                    )
        if not found:
            self.path = np.empty((0, 2))
            return self.path
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        simple = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_clear(cells[i], cells[j]):
                j -= 1
            simple.append(cells[j])
            i = j
        self.path = np.array([self._point(c) for c in simple])
        self.path[0] = start_xy
        self.path[-1] = goal_xy
        return self.path.copy()


def reference_on_path(pose, path, lookahead):
    """Look ahead along the nearest path segment, returning [x,y,tangent yaw]."""
    if len(path) < 2:
        return np.array([*pose[:2], pose[2]])
    best = None
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        ab = b - a
        norm = float(ab @ ab)
        t = np.clip((np.asarray(pose[:2]) - a) @ ab / max(norm, 1e-12), 0, 1)
        p = a + t * ab
        distance = np.linalg.norm(pose[:2] - p)
        if best is None or distance < best[0]:
            best = (distance, i, p)
    _, i, p = best
    remain = lookahead
    while i < len(path) - 1:
        vector = path[i + 1] - p
        length = np.linalg.norm(vector)
        if length >= remain and length > 1e-12:
            p = p + vector * remain / length
            return np.array([*p, np.arctan2(vector[1], vector[0])])
        remain -= length
        p = path[i + 1]
        i += 1
    vector = path[-1] - path[-2]
    return np.array([*path[-1], np.arctan2(vector[1], vector[0])])
