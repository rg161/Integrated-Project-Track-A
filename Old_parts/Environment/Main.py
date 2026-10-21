import time
import numpy as np
import mujoco
import mujoco.viewer
from Env import CustomRobotEnv

env = CustomRobotEnv(render_mode="human")
    
observation, info = env.reset()
    
for _ in range(500):
    # Sample a random command from the action space
    action = env.action_space.sample()
        
    observation, reward, terminated, truncated, info = env.step(action)
        
    if terminated or truncated:
        observation, info = env.reset()
            
env.close()

env = CustomRobotEnv(render_mode="human")
    
observation, info = env.reset()
    
for _ in range(500):
    # Sample a random command from the action space
    action = env.action_space.sample()
        
    observation, reward, terminated, truncated, info = env.step(action)
        
    if terminated or truncated:
        observation, info = env.reset()
            
env.close()