import gymnasium as gym
import numpy as np
from mujoco import viewer
from typing import Union

from gymnasium_robotics.envs.robot_env import MujocoPyRobotEnv, MujocoRobotEnv
from gymnasium_robotics.utils import rotations

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}

def _viewer_setup(env):
    lookat = env.sim.data.get_site_xpos("gripperpalm")
    for idx, value in enumerate(lookat):
        env.unwrapped.viewer.cam.lookat[idx] = value
    assert env.unwrapped.viewer is not None
    for key, value in DEFAULT_CAMERA_CONFIG.items():
        if isinstance(value, np.ndarray):
            getattr(env.unwrapped.viewer.cam, key)[:] = value
        else:
            setattr(env.unwrapped.viewer.cam, key, value)

# Define a simple random policy for demonstration
def random_policy(env):
    return env.action_space.sample()

# Define three sets of camera parameters
camera_params_front = {
    "azimuth": 0.0,
    "elevation": -14.0,
    "distance": 2.5,
    "lookat": np.array([1.3, 0.75, 0.55])
}

camera_params_right = {
    "azimuth": 60.0,
    "elevation": -14.0,
    "distance": 2.5,
    "lookat": np.array([1.3, 0.75, 0.55])
}

camera_params_left = {
    "azimuth": -60.0,
    "elevation": -14.0,
    "distance": 2.5,
    "lookat": np.array([1.3, 0.75, 0.55])
}

# Set up and run the environment with the front camera parameters
env_id = "FetchPickAndPlace-v3"

# Iterate over different camera parameters
for params in [camera_params_front, camera_params_right, camera_params_left]:
    # Initialize the environment
    env = gym.make(env_id)
    env.reset()

    # Apply camera settings
    env.unwrapped.viewer.cam.azimuth = params['azimuth']
    env.unwrapped.viewer.cam.elevation = params['elevation']
    env.unwrapped.viewer.cam.distance = params['distance']
    env.unwrapped.viewer.cam.lookat[:] = params['lookat']

    _viewer_setup(env)

    # Reset the environment to get the initial observation
    observation, info = env.reset(seed=42)

    # Run the simulation for 1000 steps with the current camera view
    for _ in range(1000):
        action = random_policy(env)  # Use the random policy to generate actions
        observation, reward, terminated, truncated, info = env.step(action)

        # Render the environment with the specified viewer settings
        env.render()

        # Check if the episode is terminated or truncated
        if terminated or truncated:
            observation, info = env.reset()

    # Close the environment to release resources
    env.close()
