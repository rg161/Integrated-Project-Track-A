"""Central parameters and reproducible scenario/layout sampling for Track A."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    """Store every parameter needed by one simulated Track A run."""

    # Timing and differential-drive geometry.
    dt: float = 0.05
    physics_dt: float = 0.002
    wheel_radius: float = 0.033
    wheel_base: float = 0.288
    wheel_limit: float = 7.88

    # Estimated-state controller and planner.
    max_v: float = 0.18
    max_omega: float = 1.2
    k1: float = 0.8
    k2: float = 2.0
    k3: float = 1.8
    robot_radius: float = 0.235
    safety_margin: float = 0.09
    grid_resolution: float = 0.06
    lookahead: float = 0.28

    # LiDAR.
    lidar_beams: int = 180
    lidar_max: float = 3.5
    lidar_sigma: float = 0.008
    lidar_period: float = 0.10
    lidar_pos: tuple = (-0.049, 0.0, 0.17)

    # Camera and charging marker.
    camera_period: float = 0.25
    camera_width: int = 480
    camera_height: int = 360
    camera_fovy: float = 58.0
    corner_sigma: float = 0.25
    marker_side: float = 0.24
    marker_id: int = 7
    marker_pos: tuple = (1.6, 0.0, 0.24)
    camera_pos: tuple = (0.078, 0.0, 0.18)
    camera_max_age: float = 0.75

    # Encoder / estimator assumptions.
    encoder_sigma: float = 0.0005
    odom_k: float = 0.0002

    # Task geometry. The start is randomised by seed inside the safe left region.
    start: tuple = (-1.45, -0.70, 0.0)
    start_x_range: tuple = (-1.75, -1.15)
    start_y_range: tuple = (-1.10, 1.10)
    goal: tuple = (1.12, 0.0, 0.0)
    bounds: tuple = (-2.2, 2.2, -1.6, 1.6)
    goal_position_tolerance: float = 0.055
    goal_heading_tolerance: float = np.deg2rad(5.0)
    max_duration: float = 100.0

    # Visual docking acceptance box.
    docking_zone_center: tuple = (1.12, 0.0)
    docking_zone_half_size: tuple = (0.12, 0.16)
    docking_zone_line_width: float = 0.012

    # Environment choice. "random" uses six seed-reproducible box obstacles.
    layout: str = "standard"
    random_obstacle_count: int = 6
    random_obstacle_x_range: tuple = (-0.80, 0.80)
    random_obstacle_y_range: tuple = (-1.15, 1.15)
    random_obstacle_half_size_range: tuple = (0.12, 0.22)

    # The second layout uses the same docking geometry but at a different
    # location on the right side of the room so the camera/marker task is
    # not identical between layouts.
    random_marker_pos: tuple = (1.60, 0.80, 0.24)
    random_goal: tuple = (1.12, 0.80, 0.0)
    random_docking_zone_center: tuple = (1.12, 0.80)

    def __post_init__(self):
        """Apply the target position associated with the selected environment layout."""
        if self.layout not in {"standard", "random"}:
            raise ValueError("Config.layout must be 'standard' or 'random'")
        if self.layout == "random":
            object.__setattr__(self, "marker_pos", self.random_marker_pos)
            object.__setattr__(self, "goal", self.random_goal)
            object.__setattr__(
                self, "docking_zone_center", self.random_docking_zone_center
            )

    @property
    def docking_zone_bounds(self):
        """Return docking rectangle bounds as ``(xmin, xmax, ymin, ymax)``."""
        cx, cy = map(float, self.docking_zone_center)
        hx, hy = map(float, self.docking_zone_half_size)
        return cx - hx, cx + hx, cy - hy, cy + hy

    def in_docking_zone(self, pose):
        """Return ``True`` when a finite pose's x/y centre lies inside the dock box."""
        p = np.asarray(pose, dtype=float).reshape(-1)
        if p.size < 2 or not np.all(np.isfinite(p[:2])):
            return False
        xmin, xmax, ymin, ymax = self.docking_zone_bounds
        return bool(xmin <= p[0] <= xmax and ymin <= p[1] <= ymax)

    @property
    def lidar_angle_min(self):
        """Return the angle of the first beam in the simulated 360-degree scan."""
        return -np.pi

    @property
    def lidar_angle_inc(self):
        """Return the angular spacing between neighbouring simulated LiDAR beams."""
        return 2.0 * np.pi / self.lidar_beams


@dataclass(frozen=True)
class Trial:
    """Describe one robustness condition applied to the otherwise identical task."""

    name: str
    noise_scale: float = 1.0
    camera_dropout: float = 0.0
    blackout: tuple = (-1.0, -1.0)
    push_force: float = 0.0
    push_window: tuple = (9.0, 9.5)


TRIALS = {
    "nominal": Trial("nominal"),
    "noisy": Trial("noisy", noise_scale=3.0, camera_dropout=0.20),
    "dropout": Trial("dropout", camera_dropout=0.15, blackout=(6.0, 14.0)),
    "disturbed": Trial("disturbed", push_force=1.5, push_window=(9.0, 9.5)),
}


def sample_start_pose(config, rng):
    """Sample a reproducible obstacle-free initial x/y pose while keeping heading fixed."""
    x = rng.uniform(*map(float, config.start_x_range))
    y = rng.uniform(*map(float, config.start_y_range))
    return np.array([x, y, float(config.start[2])], dtype=float)



def _random_layout_has_path(config, start_pose, obstacles):
    """Return True when inflated random obstacles leave an A* route to the goal.

    The check uses the same map resolution and robot clearance used by the runtime
    planner. Each obstacle is expanded by ``robot_radius + safety_margin`` and the
    room boundary is shrunk by the same amount, so a successful route represents
    a corridor that is wide enough for the robot centre to traverse safely.
    """
    xmin, xmax, ymin, ymax = map(float, config.bounds)
    res = float(config.grid_resolution)
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    clearance = float(config.robot_radius + config.safety_margin)

    xs = xmin + np.arange(nx) * res
    ys = ymin + np.arange(ny) * res
    X, Y = np.meshgrid(xs, ys)

    blocked = (
        (X <= xmin + clearance)
        | (X >= xmax - clearance)
        | (Y <= ymin + clearance)
        | (Y >= ymax - clearance)
    )
    for x, y, hx, hy, _ in obstacles:
        blocked |= (
            (np.abs(X - float(x)) <= float(hx) + clearance)
            & (np.abs(Y - float(y)) <= float(hy) + clearance)
        )

    def to_cell(xy):
        """Convert a world x/y location into a clipped grid row/column cell."""
        x, y = np.asarray(xy, dtype=float)[:2]
        ix = int(np.round((x - xmin) / res))
        iy = int(np.round((y - ymin) / res))
        return iy, ix

    start = to_cell(start_pose)
    goal = to_cell(config.goal)
    for cell in (start, goal):
        if not (0 <= cell[0] < ny and 0 <= cell[1] < nx) or blocked[cell]:
            return False

    neighbours = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    stack = [start]
    seen = {start}
    while stack:
        cy, cx = stack.pop()
        if (cy, cx) == goal:
            return True
        for dy, dx in neighbours:
            nyc, nxc = cy + dy, cx + dx
            nxt = (nyc, nxc)
            if not (0 <= nyc < ny and 0 <= nxc < nx):
                continue
            if blocked[nxt] or nxt in seen:
                continue
            if dy and dx and (blocked[cy + dy, cx] or blocked[cy, cx + dx]):
                continue
            seen.add(nxt)
            stack.append(nxt)
    return False


def _sample_candidate_obstacles(config, rng, start_pose):
    """Sample one separated set of random obstacle boxes before path validation."""
    count = int(config.random_obstacle_count)
    xmin, xmax = map(float, config.random_obstacle_x_range)
    ymin, ymax = map(float, config.random_obstacle_y_range)
    smin, smax = map(float, config.random_obstacle_half_size_range)
    start_xy = np.asarray(start_pose, dtype=float)[:2]
    goal_xy = np.asarray(config.goal, dtype=float)[:2]

    slots = np.linspace(xmin, xmax, count)
    obstacles = []
    for slot in slots:
        accepted = None
        for _ in range(250):
            hx = float(rng.uniform(smin, smax))
            hy = float(rng.uniform(smin, smax))
            x = float(np.clip(slot + rng.uniform(-0.09, 0.09), xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            centre = np.array([x, y])

            if np.linalg.norm(centre - start_xy) < 0.80:
                continue
            if np.linalg.norm(centre - goal_xy) < 0.75:
                continue

            overlaps = False
            for ox, oy, ohx, ohy, _ in obstacles:
                if abs(x - ox) < hx + ohx + 0.14 and abs(y - oy) < hy + ohy + 0.14:
                    overlaps = True
                    break
            if overlaps:
                continue

            accepted = (x, y, hx, hy, 0.20)
            break

        if accepted is None:
            return None
        obstacles.append(accepted)
    return obstacles

def sample_random_obstacles(config, rng, start_pose):
    """Create a reproducible random layout that is guaranteed to be navigable.

    Several candidate six-box layouts are sampled. A candidate is accepted only
    when the start and dock remain connected after every obstacle is inflated by
    the same robot-radius-plus-safety-margin used by the runtime A* planner. This
    prevents visually valid but practically impassable random mazes.
    """
    for _ in range(300):
        obstacles = _sample_candidate_obstacles(config, rng, start_pose)
        if obstacles is not None and _random_layout_has_path(config, start_pose, obstacles):
            return obstacles
    raise RuntimeError("Could not generate a safely navigable random obstacle layout")

