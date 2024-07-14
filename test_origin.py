import gymnasium as gym
import numpy as np

# Define the function to set up the environment with different camera viewpoints
def setup_env_with_viewpoint(env_id, render_mode="human", azimuth=132.0, elevation=-14.0, distance=2.5, lookat=None):
    if lookat is None:
        lookat = np.array([1.3, 0.75, 0.55])
    
    # Initialize the environment
    env = gym.make(env_id, render_mode=render_mode)
    env.reset()
    env.render()  # Render once to initialize the viewer
    
    # Forcefully set the viewer if not set
    if not hasattr(env.unwrapped, 'viewer') or env.unwrapped.viewer is None:
        print("Viewer not initialized, trying to reset and render again...")
        env.reset()
        env.render()
    
    # Check again if the viewer is set
    if hasattr(env.unwrapped, 'viewer') and env.unwrapped.viewer is not None:
        print("Viewer initialized, setting camera parameters...")
        env.unwrapped.viewer.cam.azimuth = azimuth
        env.unwrapped.viewer.cam.elevation = elevation
        env.unwrapped.viewer.cam.distance = distance
        env.unwrapped.viewer.cam.lookat[:] = lookat
    else:
        print("Viewer could not be initialized after reset and render.")

    return env

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

env = setup_env_with_viewpoint(env_id, **camera_params_front)

# Reset the environment to get the initial observation
observation, info = env.reset(seed=42)

# Run the simulation for 1000 steps with the current camera view
for _ in range(1000):
    action = random_policy(env)  # Use the random policy to generate actions
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if the episode is terminated or truncated
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment to release resources
env.close()
