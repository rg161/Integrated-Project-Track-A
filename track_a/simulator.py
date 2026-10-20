"""MuJoCo backend and physical sensor emulation.

This is intentionally the only runtime module (besides ground_truth/evaluation)
that accesses MuJoCo physical state.  Estimation, planning and control receive
sensor values only.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import ROOT


@dataclass
class SensorSample:
    time: float
    wheel_angles: np.ndarray
    lidar_ranges: np.ndarray


class Simulator:
    def __init__(self, config, trial, start_pose=None):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is required for the integrated simulator. Install requirements.txt "
                "inside the project virtual environment."
            ) from exc
        self.mujoco = mujoco
        self.c = config
        self.trial = trial
        scene = ROOT / "models" / "track_a_scene.xml"
        if not scene.exists():
            raise FileNotFoundError(scene)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = float(config.physics_dt)
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._last_collision = False
        self.collisions = 0

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.left_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
        self.right_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")
        self.left_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "encoder_left")
        self.right_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "encoder_right")
        self._obstacle_geom_ids = set()
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith("obstacle_") or name.startswith("wall_") or name == "charging_station":
                self._obstacle_geom_ids.add(gid)

        # Optional varied obstacle condition.
        if trial.obstacle_shift:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_a")
            if gid >= 0:
                self.model.geom_pos[gid, 1] += float(trial.obstacle_shift)

        self._configure_docking_zone_visual()
        self._set_start_pose(config.start if start_pose is None else start_pose)
        mujoco.mj_forward(self.model, self.data)
        self._substeps = max(1, int(round(config.dt / config.physics_dt)))
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[1] = 1  # static obstacles/walls/dock, excluding robot/floor/marker

    def _configure_docking_zone_visual(self):
        """Resize/reposition the four visual perimeter geoms from Config.

        The perimeter is display-only (no collision and no LiDAR return).
        """
        xmin, xmax, ymin, ymax = self.c.docking_zone_bounds
        cx, cy = self.c.docking_zone_center
        hx, hy = self.c.docking_zone_half_size
        w = float(self.c.docking_zone_line_width)
        z = 0.008
        h = 0.004
        specs = {
            "dock_zone_xmin": ((xmin, cy, z), (w, hy, h)),
            "dock_zone_xmax": ((xmax, cy, z), (w, hy, h)),
            "dock_zone_ymin": ((cx, ymin, z), (hx, w, h)),
            "dock_zone_ymax": ((cx, ymax, z), (hx, w, h)),
        }
        for name, (pos, size) in specs.items():
            gid = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if gid >= 0:
                self.model.geom_pos[gid] = np.asarray(pos, dtype=float)
                self.model.geom_size[gid] = np.asarray(size, dtype=float)

    def _set_start_pose(self, start):
        x, y, th = map(float, start)
        if self.model.nq < 7:
            raise RuntimeError("Expected a free base joint")
        self.data.qpos[:7] = [x, y, 0.0, np.cos(th / 2.0), 0.0, 0.0, np.sin(th / 2.0)]
        self.data.qvel[:6] = 0.0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def time(self):
        return float(self.data.time)

    def _encoder(self, sensor_id):
        adr = int(self.model.sensor_adr[sensor_id])
        return float(self.data.sensordata[adr])

    def _lidar_scan(self):
        mujoco = self.mujoco
        pnt = np.array(self.data.site_xpos[self.lidar_site_id], dtype=float)
        R = np.array(self.data.site_xmat[self.lidar_site_id], dtype=float).reshape(3, 3)
        angles = self.c.lidar_angle_min + self.c.lidar_angle_inc * np.arange(self.c.lidar_beams)
        out = np.full(self.c.lidar_beams, np.inf, dtype=float)
        for i, a in enumerate(angles):
            local = np.array([np.cos(a), np.sin(a), 0.0], dtype=float)
            vec = R @ local
            geomid = np.array([-1], dtype=np.int32)
            try:
                d = mujoco.mj_ray(
                    self.model,
                    self.data,
                    pnt,
                    vec,
                    self._geomgroup,
                    1,
                    -1,
                    geomid,
                )
            except TypeError:
                # Compatibility with bindings that accept a Python int for geomid output differently.
                d = mujoco.mj_ray(
                    self.model,
                    self.data,
                    pnt,
                    vec,
                    self._geomgroup,
                    1,
                    -1,
                    geomid,
                )
            if np.isfinite(d) and d >= 0.0 and d <= self.c.lidar_max:
                out[i] = float(d)
        return out

    def sample(self):
        return SensorSample(
            time=self.time,
            wheel_angles=np.array([self._encoder(self.left_sensor_id), self._encoder(self.right_sensor_id)]),
            lidar_ranges=self._lidar_scan(),
        )

    def camera_image(self):
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(
                self.model, height=self.c.camera_height, width=self.c.camera_width
            )
        self._renderer.update_scene(self.data, camera="front_camera")
        return np.array(self._renderer.render(), copy=True)

    def _apply_trial_force(self):
        self.data.xfrc_applied[:] = 0.0
        if self.trial.push_force and self.trial.push_window[0] <= self.time <= self.trial.push_window[1]:
            self.data.xfrc_applied[self.base_body_id, 1] = float(self.trial.push_force)

    def _update_collisions(self):
        colliding = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if int(con.geom1) in self._obstacle_geom_ids or int(con.geom2) in self._obstacle_geom_ids:
                # Count only if the other geom belongs to the robot (group 3).
                other = int(con.geom2) if int(con.geom1) in self._obstacle_geom_ids else int(con.geom1)
                if int(self.model.geom_group[other]) == 3:
                    colliding = True
                    break
        if colliding and not self._last_collision:
            self.collisions += 1
        self._last_collision = colliding

    def step(self, wheel_speeds):
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
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
