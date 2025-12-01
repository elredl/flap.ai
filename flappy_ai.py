
# flappy_env.py
import math
import numpy as np
import pygame
import gymnasium as gym
import random
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# from FlappyBird.src.flappy import Flappy
from FlappyBird.src.flappy import *

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
        self.game = Flappy()
        self.screen = self.game.config.screen
        self.clock = self.game.config.clock
        self.fps = self.game.config.fps
        self.flap_cooldown = 0
        self.min_flap_cooldown = 5 

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
            self.screen = pygame.display.set_mode((288, 512))
            self.clock = pygame.time.Clock()
        else:
            if not pygame.get_init():
                pygame.init()
            self.clock = pygame.time.Clock()

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
            old_score = self._get_score()
            self._update_score_from_pipes()
            new_score = self._get_score()
            if new_score > old_score:
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
        """Resets your internal game objects."""
        episode_seed = self._rng.randint(0, 2**31-1)
        pipe_rng = random.Random(episode_seed)
        self.game = Flappy()

        self.game.background = Background(self.game.config)
        self.game.floor = Floor(self.game.config)
        self.game.player = Player(self.game.config)
        self.welcome_message = WelcomeMessage(self.game.config)
        self.game_over_message = GameOver(self.game.config)
        self.game.pipes = Pipes(self.game.config, pipe_rng)
        self.game.score = Score(self.game.config)
        self.game.player.set_mode(PlayerMode.NORMAL)

        self.player = self.game.player
        self.pipes = self.game.pipes
        self.score = 0


    def _do_action(self, action: int):
        """Maps action to your game input (flap or no-op)."""
        if self.flap_cooldown > 0:
            self.flap_cooldown -= 1
            return

        if action == 1:
            self.game.player.flap()
            self.flap_cooldown = self.min_flap_cooldown

    def _tick_game(self):
        """Advance one frame (no human events during training)."""
        self.game.background.tick()
        self.game.floor.tick()
        self.game.pipes.tick()
        self.game.score.tick()
        self.game.player.tick()

        if self.render_mode == "human":
            pygame.display.update()
        self.game.config.tick()

    def _update_score_from_pipes(self):
        for pipe in self.game.pipes.upper:
            if self.game.player.crossed(pipe):
                self.game.score.add()

    def _is_dead(self) -> bool:
        return self.game.player.collided(self.game.pipes, self.game.floor)

    def _get_score(self) -> int:
        return self.game.score.score

    def _next_pipe(self):
        """Return the first pipe ahead of the player."""
        player_x = self.player.x
        future_pipes = [p for p in self.pipes.upper if p.x + p.w > player_x]
        if not future_pipes:
            return None, None
        upper = min(future_pipes, key=lambda p: p.x)
        idx = self.pipes.upper.index(upper)
        lower = self.pipes.lower[idx]

        upper_bottom = upper.y + upper.h
        lower_top = lower.y
        gap_y = (upper_bottom - lower_top)/2
        return upper, gap_y

    def _get_obs(self) -> np.ndarray:
        """Pull raw values from your objects"""
        px = self.player.cx
        py = self.player.y
        vy = self.player.vel_y

        pipe, gap_y = self._next_pipe()
        if pipe is None:
            dx = 0.0
            dy = 0.0
            vx = 0.0
        else:
            dx = pipe.cx - px
            vx = pipe.vel_x
            dy = gap_y - py

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
