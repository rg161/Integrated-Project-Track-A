"""MuJoCo boundary: actuators and simulated sensor acquisition only.

Access to simulator state here emulates encoders, optical rays and collision
contacts. Neither estimator nor controller receives an MjData/MjModel object.
"""

from dataclasses import dataclass
import numpy as np
import mujoco
from .scene import build_scene


@dataclass(frozen=True)
class SensorPacket:
    time: float
    wheel_angles: np.ndarray
    ranges: np.ndarray | None
    image: np.ndarray | None


class Simulator:
    def __init__(self, config, trial, seed=1, viewer=False):
        self.config, self.trial = config, trial
        # Separate streams mean camera dropouts do not alter encoder/lidar noise.
        seq = np.random.SeedSequence(seed).spawn(3)
        self.encoder_rng, self.lidar_rng, self.camera_rng = [
            np.random.default_rng(s) for s in seq
        ]
        self.model = mujoco.MjModel.from_xml_string(build_scene(config, trial))
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base").id
        self.angles = np.linspace(-np.pi, np.pi, config.lidar_beams, endpoint=False)
        self._directions = np.c_[
            np.cos(self.angles), np.sin(self.angles), np.zeros_like(self.angles)
        ]
        self._ray_groups = np.array(
            [1, 0, 0, 0, 0, 0], dtype=np.uint8
        )  # scene obstacles only
        self._geomid = np.zeros(1, dtype=np.int32)
        self._renderer = None
        self.viewer = None
        self._next_lidar, self._next_camera = 0.0, 0.0
        self.collisions = 0
        self._touching = False
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)  # settle on floor with zero commands
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        if viewer:
            import mujoco.viewer as mjviewer

            self.viewer = mjviewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0, 0, 0.2]
            self.viewer.cam.distance = 5.8
            self.viewer.cam.azimuth = 125
            self.viewer.cam.elevation = -65

    @property
    def time(self):
        return float(self.data.time)

    def _lidar(self):
        site = self.data.site("lidar")
        origin = site.xpos.copy()
        rotation = site.xmat.reshape(3, 3)
        ranges = np.empty(len(self.angles))
        for i, direction in enumerate(self._directions @ rotation.T):
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                self._ray_groups,
                True,
                self.base_id,
                self._geomid,
            )
            ranges[i] = distance if 0 <= distance <= self.config.lidar_max else np.inf
        hit = np.isfinite(ranges)
        ranges[hit] += self.lidar_rng.normal(
            0, self.config.lidar_sigma * self.trial.noise_scale, hit.sum()
        )
        return ranges

    def camera_image(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.camera_height,
                width=self.config.camera_width,
            )
        self._renderer.update_scene(self.data, camera="front_camera")
        return self._renderer.render().copy()

    def sample(self):
        wheel_angles = np.array(
            [
                self.data.sensor("encoder_left").data[0],
                self.data.sensor("encoder_right").data[0],
            ]
        )
        wheel_angles += self.encoder_rng.normal(
            0, self.config.encoder_sigma * self.trial.noise_scale, 2
        )
        ranges, image = None, None
        if self.time + 1e-9 >= self._next_lidar:
            ranges = self._lidar()
            self._next_lidar += self.config.lidar_period
        if self.time + 1e-9 >= self._next_camera:
            self._next_camera += self.config.camera_period
            if not (self.trial.blackout[0] <= self.time <= self.trial.blackout[1]):
                if self.camera_rng.random() >= self.trial.camera_dropout:
                    image = self.camera_image()
        return SensorPacket(self.time, wheel_angles, ranges, image)

    def step(self, wheels):
        if np.shape(wheels) != (2,) or not np.all(np.isfinite(wheels)):
            raise ValueError("Expected two finite wheel commands")
        self.data.ctrl[:] = np.clip(
            wheels, -self.config.wheel_limit, self.config.wheel_limit
        )
        for _ in range(round(self.config.dt / self.config.physics_dt)):
            self.data.xfrc_applied[:] = 0
            if self.trial.push_window[0] <= self.time < self.trial.push_window[1]:
                self.data.xfrc_applied[self.base_id, 1] = self.trial.push_force
            mujoco.mj_step(self.model, self.data)
            touching = False
            for contact in self.data.contact:
                g1, g2 = contact.geom1, contact.geom2
                bodies = [self.model.geom_bodyid[g1], self.model.geom_bodyid[g2]]
                # Obstacles/walls are group 0; ground and marker visuals are excluded.
                if (
                    bodies[0] != 0 and bodies[1] == 0 and self.model.geom_group[g2] == 0
                ) or (
                    bodies[1] != 0 and bodies[0] == 0 and self.model.geom_group[g1] == 0
                ):
                    touching = True
            if touching and not self._touching:
                self.collisions += 1
            self._touching = touching
        # mj_step leaves some derived positions at the previous integration stage.
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        self.data.ctrl[:] = 0
        if self._renderer is not None:
            self._renderer.close()
        if self.viewer is not None:
            self.viewer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
