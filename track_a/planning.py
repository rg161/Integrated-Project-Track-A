"""LiDAR occupancy mapping and grid A* planning using estimated robot pose."""

from heapq import heappop, heappush
import numpy as np
from .noise import clean_scan


class LidarPlanner:
    def __init__(self, config):
        self.config = config
        self.xmin, self.xmax, self.ymin, self.ymax = map(float, config.bounds)
        self.res = float(config.grid_resolution)
        self.nx = int(np.ceil((self.xmax - self.xmin) / self.res)) + 1
        self.ny = int(np.ceil((self.ymax - self.ymin) / self.res)) + 1
        self.log_odds = np.zeros((self.ny, self.nx), dtype=float)
        self.angles = config.lidar_angle_min + config.lidar_angle_inc * np.arange(config.lidar_beams)
        self.last_path = np.empty((0, 2), dtype=float)
        self._inflation_cells = int(np.ceil((config.robot_radius + config.safety_margin) / self.res))
        self._disk_offsets = self._make_disk(self._inflation_cells)

    @staticmethod
    def _make_disk(radius):
        out = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    out.append((dy, dx))
        return out

    def world_to_cell(self, xy):
        x, y = np.asarray(xy, dtype=float)[:2]
        ix = int(np.round((x - self.xmin) / self.res))
        iy = int(np.round((y - self.ymin) / self.res))
        return iy, ix

    def cell_to_world(self, cell):
        iy, ix = cell
        return np.array([self.xmin + ix * self.res, self.ymin + iy * self.res], dtype=float)

    def _inside(self, cell):
        iy, ix = cell
        return 0 <= iy < self.ny and 0 <= ix < self.nx

    def _ray_cells(self, p0, p1):
        """Sample a world-space ray densely enough to cover every grid cell."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(dist / (0.45 * self.res))) + 1)
        pts = p0[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0)[None, :]
        cells = []
        last = None
        for p in pts:
            c = self.world_to_cell(p)
            if c != last and self._inside(c):
                cells.append(c)
                last = c
        return cells

    def update(self, pose, ranges):
        """Update occupancy grid from a LiDAR scan and estimated base pose."""
        pose = np.asarray(pose, dtype=float).reshape(3)
        r = np.asarray(ranges, dtype=float).reshape(-1)
        if len(r) != self.config.lidar_beams:
            raise ValueError("LiDAR scan length does not match Config.lidar_beams")
        # Preserve infinity as no-return; reject actual invalid/too-close values.
        finite = np.isfinite(r)
        valid = finite & (r >= 0.12) & (r <= self.config.lidar_max)
        if not np.any(valid):
            return False

        c, s = np.cos(pose[2]), np.sin(pose[2])
        lidar_offset = np.asarray(self.config.lidar_pos, dtype=float)[:2]
        origin = pose[:2] + np.array([c * lidar_offset[0] - s * lidar_offset[1], s * lidar_offset[0] + c * lidar_offset[1]])

        # Decay old evidence slowly so occasional spurious returns do not become permanent.
        self.log_odds *= 0.997
        for i in np.flatnonzero(valid):
            ang = pose[2] + self.angles[i]
            end = origin + r[i] * np.array([np.cos(ang), np.sin(ang)])
            cells = self._ray_cells(origin, end)
            if not cells:
                continue
            for cell in cells[:-1]:
                self.log_odds[cell] = np.clip(self.log_odds[cell] - 0.20, -4.0, 4.0)
            self.log_odds[cells[-1]] = np.clip(self.log_odds[cells[-1]] + 0.85, -4.0, 4.0)
        return True

    def occupied(self):
        return self.log_odds > 0.55

    def inflated_occupied(self):
        occ = self.occupied()
        inf = occ.copy()
        ys, xs = np.nonzero(occ)
        for y, x in zip(ys, xs):
            for dy, dx in self._disk_offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < self.ny and 0 <= xx < self.nx:
                    inf[yy, xx] = True
        # Bounds themselves are impassable.
        inf[0, :] = inf[-1, :] = True
        inf[:, 0] = inf[:, -1] = True
        return inf

    @staticmethod
    def _nearest_free(cell, blocked):
        if not blocked[cell]:
            return cell
        ny, nx = blocked.shape
        cy, cx = cell
        for rad in range(1, max(nx, ny)):
            candidates = []
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if max(abs(dx), abs(dy)) != rad:
                        continue
                    y, x = cy + dy, cx + dx
                    if 0 <= y < ny and 0 <= x < nx and not blocked[y, x]:
                        candidates.append((dx * dx + dy * dy, (y, x)))
            if candidates:
                return min(candidates)[1]
        return None

    def plan(self, start_xy, goal_xy):
        start = self.world_to_cell(start_xy)
        goal = self.world_to_cell(goal_xy)
        if not (self._inside(start) and self._inside(goal)):
            self.last_path = np.empty((0, 2))
            return self.last_path.copy()
        blocked = self.inflated_occupied()
        start = self._nearest_free(start, blocked)
        goal = self._nearest_free(goal, blocked)
        if start is None or goal is None:
            self.last_path = np.empty((0, 2))
            return self.last_path.copy()

        neighbours = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
        ]
        pq = [(0.0, start)]
        came = {start: None}
        g = {start: 0.0}
        while pq:
            _, cur = heappop(pq)
            if cur == goal:
                break
            for dy, dx, cost in neighbours:
                nxt = (cur[0] + dy, cur[1] + dx)
                if not self._inside(nxt) or blocked[nxt]:
                    continue
                # Prevent diagonal corner-cutting through occupied orthogonal cells.
                if dy and dx and (blocked[cur[0] + dy, cur[1]] or blocked[cur[0], cur[1] + dx]):
                    continue
                ng = g[cur] + cost
                if ng < g.get(nxt, np.inf):
                    g[nxt] = ng
                    came[nxt] = cur
                    h = np.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                    heappush(pq, (ng + h, nxt))
        if goal not in came:
            self.last_path = np.empty((0, 2))
            return self.last_path.copy()
        cells = []
        cur = goal
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        cells.reverse()
        path = np.array([self.cell_to_world(c) for c in cells], dtype=float)
        # Replace quantised endpoints with requested endpoints.
        path[0] = np.asarray(start_xy, dtype=float)[:2]
        path[-1] = np.asarray(goal_xy, dtype=float)[:2]
        path = self._simplify(path, blocked)
        self.last_path = path
        return path.copy()

    def _line_free(self, a, b, blocked):
        cells = self._ray_cells(a, b)
        return bool(cells) and all(not blocked[c] for c in cells)

    def _simplify(self, path, blocked):
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_free(path[i], path[j], blocked):
                j -= 1
            out.append(path[j])
            i = j
        return np.asarray(out)

    def save(self, path):
        np.savez_compressed(path, log_odds=self.log_odds, bounds=np.array(self.config.bounds), resolution=self.res)
