
# flappy_env.py
import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# environment: flappy bird
# state: player (y, y_vel), pipe(x, bottom-top-y, top-bottom-y), score
# actions: flap, no_flap
# reward: time alive/score (synonimous)
# return: if next pipe is higher/lower than current (?)
# episode: 100

# very very simple skeleton code from gpt to understand gymnasium structure

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, frame_skip: int = 2, max_steps: int = 20000):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self._rng = np.random.RandomState(0)

        # ---- ACTION & OBS SPACES ----
        # 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation: [player_y, player_vy, dx_to_pipe, dy_to_gap, pipe_vx]
        high = np.array([1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # ---- Pygame/Game handles (fill these in) ----
        self.game = None
        self.screen = None
        self.clock = None

        # Caches
        self.player = None
        self.pipes = None
        self.score = 0
        self.steps = 0

    # ---------------- Core RL API ----------------
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.seed(seed)

        # Initialize/Reset your Pygame game here
        if self.render_mode == "human":
            if not pygame.get_init():
                pygame.init()
            self.screen = pygame.display.set_mode((288, 512))  # replace with your game size
            self.clock = pygame.time.Clock()
        else:
            if not pygame.get_init():
                pygame.init()
            self.clock = pygame.time.Clock()

        # --- TODO: call into your game’s reset/new_game ---
        # Example placeholders:
        # self.game = Game()
        # self.game.reset()
        # self.player = self.game.player
        # self.pipes = self.game.pipes
        self._new_game()

        self.score = 0
        self.steps = 0
        obs = self._get_obs()
        info = {"score": self.score}
        return obs, info

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False

        # Apply action for frame_skip frames
        for _ in range(self.frame_skip):
            self._do_action(action)
            # --- advance game one tick ---
            self._tick_game()

            # Reward for pipe passes; compute delta if game tracks score
            new_score = self._get_score()
            if new_score > self.score:
                reward += (new_score - self.score) * 1.0
                self.score = new_score

            # Small living reward (optional)
            reward += 0.01

            if self._is_dead():
                reward -= 1.0
                terminated = True
                break

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"score": self.score}
        if self.render_mode == "human":
            self.render()
            # keep FPS roughly consistent for human view
            self.clock.tick(self.metadata["render_fps"])

        return obs, reward, terminated, truncated, info

    def render(self):
        # If your game already renders during tick, you can no-op here
        # Or blit to self.screen if you manage drawing yourself.
        pygame.display.flip()

    def close(self):
        try:
            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass

    # ---------------- Game glue (replace TODOs) ----------------
    def _new_game(self):
        """Reset your internal game objects here."""
        # TODO: wire to your code
        # eg: self.game = Game(seed=self._rng.randint(0, 10_000))
        # self.player = self.game.player
        # self.pipes = self.game.pipes
        pass

    def _do_action(self, action: int):
        """Map action to your game input (flap or no-op)."""
        # TODO:
        # if action == 1:
        #     self.game.flap()
        pass

    def _tick_game(self):
        """Advance one frame (no human events during training)."""
        # TODO:
        # self.game.update()  # physics, spawn, collisions
        # self.game.draw(self.screen)  # only if rendering
        pass

    def _is_dead(self) -> bool:
        # TODO: return True if collision or out-of-bounds
        # Example:
        # return self.game.collided or self.player.y < 0 or self.player.y > FLOOR_Y
        return False

    def _get_score(self) -> int:
        # TODO: return your game’s score
        return self.score

    def _next_pipe(self):
        """Return the first pipe ahead of the player."""
        # TODO: pick the pipe with x > player.x with smallest x
        # return next_pipe, gap_center_y
        return None, None

    def _get_obs(self) -> np.ndarray:
        # Pull raw values from your objects
        # --- TODO: fetch from your actual objects ---
        # px, py = self.player.x, self.player.y
        # vy = self.player.vy
        # pipe, gap_y = self._next_pipe()
        # dx = pipe.x - px
        # vx = pipe.vx
        # Compose and normalize:
        py = 0.0
        vy = 0.0
        dx = 0.0
        dy = 0.0
        vx = 0.0

        obs = np.array([py, vy, dx, dy, vx], dtype=np.float32)
        return self._normalize(obs)

    # ---------------- Normalization helpers ----------------
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        # Replace with real scales from your game dimensions/physics
        # Example scales:
        # y in [0, 512]; vy in ~[-10, 10]; dx in [0, 300]; dy in [-300, 300]; vx in [-5, 5]
        scales = np.array([512.0, 10.0, 300.0, 300.0, 5.0], dtype=np.float32)
        # shift to roughly zero-centered when appropriate (dy, vy already centered)
        centered = obs.copy()
        centered[0] = (centered[0] - 256.0)  # center y
        centered[2] = (centered[2] - 150.0)  # center dx roughly
        return np.clip(centered / scales, -1.0, 1.0)
