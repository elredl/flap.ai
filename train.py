from flappy_ai import FlappyBirdEnv
import numpy as np

env = FlappyBirdEnv(render_mode="human")
obs, info = env.reset(seed=0)

for _ in range(2000):
    action = env.action_space.sample()  # random for testing
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward)
    if terminated or truncated:
        break

env.close()
