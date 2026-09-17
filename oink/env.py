"""
Phase 2 — the pig environment.

A 2D Pygame/Gymnasium environment: a pig sprite at the bottom of the screen
steers left/right to dodge objects falling from above. Optomotor circuits in
flies exist to compute self-motion/relative-motion from visual flow, so
"dodge the falling object using only its retinotopic position" is a
reasonable toy analogue of the real task the circuit evolved for.
"""
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402  (must follow the SDL_VIDEODRIVER default above)

WIDTH, HEIGHT = 400, 300
PIG_Y = HEIGHT - 30
PIG_HALF_WIDTH = 20
PIG_STEP = 8
STIMULUS_RADIUS = 8
MAX_STEPS = 500

PINK = (230, 150, 170)
DARK_PINK = (200, 110, 130)
RED = (200, 40, 40)
SKY = (235, 245, 250)
GROUND = (210, 230, 190)


class PigDodgeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=left, 1=stay, 2=right

        self._surface = pygame.Surface((WIDTH, HEIGHT))
        self._rng = np.random.default_rng()

        self.pig_x = WIDTH / 2
        self.stim_x = 0.0
        self.stim_y = 0.0
        self.stim_vx = 0.0
        self.stim_vy = 0.0
        self.steps = 0

    def _spawn_stimulus(self):
        self.stim_x = self._rng.uniform(STIMULUS_RADIUS, WIDTH - STIMULUS_RADIUS)
        self.stim_y = 0.0
        self.stim_vx = self._rng.uniform(-1.5, 1.5)
        self.stim_vy = self._rng.uniform(3.0, 5.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pig_x = WIDTH / 2
        self._spawn_stimulus()
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        return np.array(
            [
                (self.pig_x / WIDTH) * 2 - 1,
                (self.stim_x / WIDTH) * 2 - 1,
                (self.stim_y / HEIGHT) * 2 - 1,
                np.clip(self.stim_vx / 2.0, -1, 1),
                np.clip(self.stim_vy / 6.0, -1, 1),
            ],
            dtype=np.float32,
        )

    def step(self, action: int):
        if action == 0:
            self.pig_x -= PIG_STEP
        elif action == 2:
            self.pig_x += PIG_STEP
        self.pig_x = float(np.clip(self.pig_x, PIG_HALF_WIDTH, WIDTH - PIG_HALF_WIDTH))

        self.stim_x += self.stim_vx
        self.stim_y += self.stim_vy
        self.stim_x = float(np.clip(self.stim_x, STIMULUS_RADIUS, WIDTH - STIMULUS_RADIUS))

        self.steps += 1
        reward = 0.05
        terminated = False
        truncated = False

        reached_pig_height = self.stim_y >= PIG_Y - STIMULUS_RADIUS
        if reached_pig_height:
            hit = abs(self.stim_x - self.pig_x) <= (PIG_HALF_WIDTH + STIMULUS_RADIUS)
            if hit:
                reward = -10.0
                terminated = True
            else:
                reward = 1.0
                self._spawn_stimulus()

        if not terminated and self.steps >= MAX_STEPS:
            truncated = True

        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        surf = self._surface
        surf.fill(SKY)
        pygame.draw.rect(surf, GROUND, (0, PIG_Y + 15, WIDTH, HEIGHT - PIG_Y - 15))

        px, py = int(self.pig_x), PIG_Y
        pygame.draw.ellipse(surf, PINK, (px - 22, py - 14, 44, 28))
        pygame.draw.polygon(surf, DARK_PINK, [(px - 18, py - 12), (px - 24, py - 22), (px - 10, py - 14)])
        pygame.draw.polygon(surf, DARK_PINK, [(px + 18, py - 12), (px + 24, py - 22), (px + 10, py - 14)])
        pygame.draw.circle(surf, DARK_PINK, (px + 20, py), 8)
        pygame.draw.circle(surf, (80, 40, 50), (px + 23, py - 2), 1)
        pygame.draw.circle(surf, (80, 40, 50), (px + 23, py + 2), 1)
        pygame.draw.circle(surf, (30, 30, 30), (px + 5, py - 6), 2)

        pygame.draw.circle(surf, RED, (int(self.stim_x), int(self.stim_y)), STIMULUS_RADIUS)

        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    def close(self):
        pass
