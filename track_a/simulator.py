"""MuJoCo simulation backend that exposes only simulated sensor readings and actuation."""

from dataclasses import dataclass
import numpy as np

from .config import ROOT, sample_random_obstacles


@dataclass
class SensorSample:
    """Bundle one simulator timestamp with wheel-encoder and LiDAR measurements."""

    time: float
    wheel_angles: np.ndarray
    lidar_ranges: np.ndarray


class Simulator:
    """Load the TurtleBot3 scene, generate sensors, apply wheel commands and count collisions."""

    def __init__(self, config, trial, seed, start_pose):
        """Create one MuJoCo run and configure the selected obstacle layout and start pose."""
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is required. Activate the project environment and run "
                "'pip install -r requirements.txt'."
            ) from exc

        self.mujoco = mujoco
        self.c = config
        self.trial = trial
        self.seed = int(seed)
        self.start_pose = np.asarray(start_pose, dtype=float).reshape(3)

        scene = ROOT / "models" / "track_a_scene.xml"
        if not scene.exists():
            raise FileNotFoundError(scene)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = float(config.physics_dt)
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._last_collision = False
        self.collisions = 0

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.lidar_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar"
        )
        self.left_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left"
        )
        self.right_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right"
        )
        self.left_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "encoder_left"
        )
        self.right_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "encoder_right"
        )

        self._configure_target_visuals()
        self._configure_layout()
        self._configure_docking_zone_visual()
        self._set_start_pose(self.start_pose)
        mujoco.mj_forward(self.model, self.data)

        self._obstacle_geom_ids = set()
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith("obstacle_") or name.startswith("random_obstacle_") or name.startswith("wall_") or name == "charging_station":
                self._obstacle_geom_ids.add(gid)

        self._substeps = max(1, int(round(config.dt / config.physics_dt)))
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[1] = 1

    def _geom_id(self, name):
        """Return a MuJoCo geom ID by name, or ``-1`` when the geom is absent."""
        return self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name
        )

    def _hide_geom(self, name):
        """Move an unused static obstacle far outside the room so it cannot be sensed or hit."""
        gid = self._geom_id(name)
        if gid >= 0:
            self.model.geom_pos[gid] = np.array([50.0, 50.0, 0.20])

    def _set_box_geom(self, name, spec):
        """Set one world box geom from ``(x, y, half_x, half_y, half_z)``."""
        gid = self._geom_id(name)
        if gid < 0:
            raise RuntimeError(f"Scene is missing expected geom '{name}'")
        x, y, hx, hy, hz = map(float, spec)
        self.model.geom_pos[gid] = np.array([x, y, hz])
        self.model.geom_size[gid] = np.array([hx, hy, hz])

    def _configure_target_visuals(self):
        """Move the charging station and rendered ArUco board to the configured marker location."""
        base_marker = np.array([1.60, 0.0, 0.24], dtype=float)
        marker_target = np.asarray(self.c.marker_pos, dtype=float).reshape(3)
        delta = marker_target - base_marker

        # Shift every piece of the rendered marker by the same world offset so
        # the visual board and the pose used by sensing.py always agree.
        for gid in range(self.model.ngeom):
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, gid
            ) or ""
            if name.startswith("aruco_"):
                self.model.geom_pos[gid] = self.model.geom_pos[gid] + delta

        # The charging station sits 0.12 m behind the marker along world +x.
        station_id = self._geom_id("charging_station")
        if station_id >= 0:
            self.model.geom_pos[station_id] = np.array(
                [marker_target[0] + 0.12, marker_target[1], 0.18 + delta[2]],
                dtype=float,
            )

    def _configure_layout(self):
        """Select the standard two-obstacle room or configure six seed-randomised obstacles."""
        random_names = [
            f"random_obstacle_{i + 1}" for i in range(self.c.random_obstacle_count)
        ]

        if self.c.layout == "standard":
            for name in random_names:
                self._hide_geom(name)
            self.obstacle_layout = [
                {"name": "obstacle_a", "x": -0.25, "y": -0.31, "half_x": 0.20, "half_y": 0.28},
                {"name": "obstacle_b", "x": 0.42, "y": 0.72, "half_x": 0.24, "half_y": 0.18},
            ]
            return

        if self.c.layout != "random":
            raise ValueError("Config.layout must be 'standard' or 'random'")

        self._hide_geom("obstacle_a")
        self._hide_geom("obstacle_b")
        rng = np.random.default_rng(self.seed * 10 + 6)
        specs = sample_random_obstacles(self.c, rng, self.start_pose)
        self.obstacle_layout = []
        for name, spec in zip(random_names, specs):
            self._set_box_geom(name, spec)
            x, y, hx, hy, hz = spec
            self.obstacle_layout.append(
                {
                    "name": name,
                    "x": float(x),
                    "y": float(y),
                    "half_x": float(hx),
                    "half_y": float(hy),
                    "half_z": float(hz),
                }
            )

    def _configure_docking_zone_visual(self):
        """Resize and reposition the four display-only charging-zone perimeter geoms."""
        xmin, xmax, ymin, ymax = self.c.docking_zone_bounds
        cx, cy = self.c.docking_zone_center
        hx, hy = self.c.docking_zone_half_size
        width = float(self.c.docking_zone_line_width)
        z = 0.008
        half_height = 0.004
        specs = {
            "dock_zone_xmin": ((xmin, cy, z), (width, hy, half_height)),
            "dock_zone_xmax": ((xmax, cy, z), (width, hy, half_height)),
            "dock_zone_ymin": ((cx, ymin, z), (hx, width, half_height)),
            "dock_zone_ymax": ((cx, ymax, z), (hx, width, half_height)),
        }
        for name, (pos, size) in specs.items():
            gid = self._geom_id(name)
            if gid >= 0:
                self.model.geom_pos[gid] = np.asarray(pos, dtype=float)
                self.model.geom_size[gid] = np.asarray(size, dtype=float)

    def _set_start_pose(self, start):
        """Place the free-base robot at the requested planar start pose with zero velocity."""
        x, y, th = map(float, start)
        if self.model.nq < 7:
            raise RuntimeError("Expected TurtleBot3 model to have a free base joint")
        self.data.qpos[:7] = [
            x,
            y,
            0.0,
            np.cos(th / 2.0),
            0.0,
            0.0,
            np.sin(th / 2.0),
        ]
        self.data.qvel[:6] = 0.0

    def __enter__(self):
        """Return this simulator when used as a context manager."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close rendering resources when leaving the simulator context."""
        self.close()
        return False

    @property
    def time(self):
        """Return current MuJoCo simulation time in seconds."""
        return float(self.data.time)

    def _encoder(self, sensor_id):
        """Read one joint-position sensor and return its wheel angle in radians."""
        address = int(self.model.sensor_adr[sensor_id])
        return float(self.data.sensordata[address])

    def _lidar_scan(self):
        """Ray-cast the configured 360-degree LiDAR and return one range per beam."""
        pnt = np.array(self.data.site_xpos[self.lidar_site_id], dtype=float)
        R = np.array(self.data.site_xmat[self.lidar_site_id], dtype=float).reshape(3, 3)
        angles = self.c.lidar_angle_min + self.c.lidar_angle_inc * np.arange(
            self.c.lidar_beams
        )
        out = np.full(self.c.lidar_beams, np.inf, dtype=float)
        for i, angle in enumerate(angles):
            local = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
            direction = R @ local
            geomid = np.array([-1], dtype=np.int32)
            distance = self.mujoco.mj_ray(
                self.model,
                self.data,
                pnt,
                direction,
                self._geomgroup,
                1,
                -1,
                geomid,
            )
            if np.isfinite(distance) and 0.0 <= distance <= self.c.lidar_max:
                out[i] = float(distance)
        return out

    def sample(self):
        """Return the current time, wheel angles and raw LiDAR scan."""
        return SensorSample(
            time=self.time,
            wheel_angles=np.array(
                [self._encoder(self.left_sensor_id), self._encoder(self.right_sensor_id)]
            ),
            lidar_ranges=self._lidar_scan(),
        )

    def camera_image(self):
        """Render and return the RGB image from the TurtleBot3 front camera."""
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(
                self.model,
                height=self.c.camera_height,
                width=self.c.camera_width,
            )
        self._renderer.update_scene(self.data, camera="front_camera")
        return np.array(self._renderer.render(), copy=True)

    def _apply_trial_force(self):
        """Apply the configured brief sideways disturbance for the disturbed scenario."""
        self.data.xfrc_applied[:] = 0.0
        if (
            self.trial.push_force
            and self.trial.push_window[0] <= self.time <= self.trial.push_window[1]
        ):
            self.data.xfrc_applied[self.base_body_id, 1] = float(self.trial.push_force)

    def _update_collisions(self):
        """Count new robot contacts with walls, obstacles or the charging station."""
        colliding = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self._obstacle_geom_ids or g2 in self._obstacle_geom_ids:
                other = g2 if g1 in self._obstacle_geom_ids else g1
                if int(self.model.geom_group[other]) == 3:
                    colliding = True
                    break
        if colliding and not self._last_collision:
            self.collisions += 1
        self._last_collision = colliding

    def step(self, wheel_speeds):
        """Apply left/right wheel-speed commands and advance one controller timestep."""
        wheels = np.asarray(wheel_speeds, dtype=float).reshape(2)
        wheels = np.clip(wheels, -self.c.wheel_limit, self.c.wheel_limit)
        self.data.ctrl[self.left_act_id] = wheels[0]
        self.data.ctrl[self.right_act_id] = wheels[1]
        for _ in range(self._substeps):
            self._apply_trial_force()
            self.mujoco.mj_step(self.model, self.data)
            self._update_collisions()
        return self.time

    def close(self):
        """Release the off-screen renderer when one was created for camera sensing."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
