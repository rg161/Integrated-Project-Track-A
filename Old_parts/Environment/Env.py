import numpy as np
import os
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class CustomRobotEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"], 
        "render_fps": 100, 
    }

    def __init__(self, **kwargs):
        xml_path = os.path.abspath("Integrated-Project-Track-A/Scene.xml")

        super().__init__(
            model_path = xml_path, 
            frame_skip = 5, 
            observation_space = None, 
            **kwargs
        )
        gym.utils.EzPickle.__init__(self, **kwargs)

    def step(self, action):

        self.do_simulation(action, self.frame_skip)

        robot_pos = self.data.geom("robot_geom").xpos
        charger_pos = self.data.geom("charging_station").xpos

        distance = np.linalg.norm(robot_pos - charger_pos)
        reward = -distance

        terminated = bool(distance < 0.25)
        truncated = False

        observation = self._get_obs()
        info = {"distance_to_charger": distance}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat, 
            self.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low = -0.1, high = 0.1, size = self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low = -0.1, high = 0.1, size = self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()