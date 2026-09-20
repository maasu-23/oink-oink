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
MAX_STIM_VX = 2.0
MAX_STEPS = 500

PINK = (230, 150, 170)
DARK_PINK = (200, 110, 130)
RED = (200, 40, 40)
SKY = (235, 245, 250)
GROUND = (210, 230, 190)

# Blocky side-view pig, facing right. Each char is a 2x2 pixel block; the
# 20x14 grid scales to 40x28 px, matching the PIG_HALF_WIDTH hitbox.
PIXEL = 2
PIG_PALETTE = {"p": PINK, "d": DARK_PINK, "n": (150, 70, 90), "k": (30, 30, 30)}
PIG_BODY = [
    "....................",
    "............pppppppp",
    "............pppppppp",
    "..ppppppppppppppkppp",
    "..pppppppppppppppppp",
    "..pppppppppppppppddd",
    "..pppppppppppppppdnd",
    "..pppppppppppppppddd",
    "..pppppppppppppppppp",
    "..pppppppppppppppppp",
]
PIG_LEGS = [
    [  # standing / stride A
        "...pp..pp..pp..pp...",
        "...pp..pp..pp..pp...",
        "...pp..pp..pp..pp...",
        "...dd..dd..dd..dd...",
    ],
    [  # stride B: legs 1 and 3 lifted
        "...pp..pp..pp..pp...",
        "...pp..pp..pp..pp...",
        "...dd..pp..dd..pp...",
        ".......dd......dd...",
    ],
]


def _build_pig_sprite(legs: list[str]) -> pygame.Surface:
    rows = PIG_BODY + legs
    surf = pygame.Surface((len(rows[0]) * PIXEL, len(rows) * PIXEL), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in PIG_PALETTE:
                surf.fill(PIG_PALETTE[ch], (x * PIXEL, y * PIXEL, PIXEL, PIXEL))
    return surf


class PigDodgeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=left, 1=stay, 2=right

        self._surface = pygame.Surface((WIDTH, HEIGHT))
        self._pig_frames = [_build_pig_sprite(legs) for legs in PIG_LEGS]
        self._rng = np.random.default_rng()

        self.pig_x = WIDTH / 2
        self.stim_x = 0.0
        self.stim_y = 0.0
        self.stim_vx = 0.0
        self.stim_vy = 0.0
        self.steps = 0
        # Render-only state (never observed by the agent).
        self.facing = 1
        self.moving = False
        self.stride = 0

    def _spawn_stimulus(self):
        # Thrown *at* the pig: spawn near it and aim at (roughly) where it is
        # now. With uniform random spawns, parking in a corner dodged ~90% of
        # balls by luck and the policy learned to camp instead of dodge.
        self.stim_x = float(np.clip(self.pig_x + self._rng.normal(0, 80), STIMULUS_RADIUS, WIDTH - STIMULUS_RADIUS))
        self.stim_y = 0.0
        self.stim_vy = self._rng.uniform(3.0, 5.0)
        target_x = self.pig_x + self._rng.normal(0, 15)
        steps_to_arrive = (PIG_Y - STIMULUS_RADIUS) / self.stim_vy
        self.stim_vx = float(np.clip((target_x - self.stim_x) / steps_to_arrive, -MAX_STIM_VX, MAX_STIM_VX))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pig_x = WIDTH / 2
        self._spawn_stimulus()
        self.steps = 0
        self.facing, self.moving, self.stride = 1, False, 0
        return self._obs(), {}

    def _obs(self):
        return np.array(
            [
                (self.pig_x / WIDTH) * 2 - 1,
                (self.stim_x / WIDTH) * 2 - 1,
                (self.stim_y / HEIGHT) * 2 - 1,
                np.clip(self.stim_vx / MAX_STIM_VX, -1, 1),
                np.clip(self.stim_vy / 6.0, -1, 1),
            ],
            dtype=np.float32,
        )

    def step(self, action: int):
        if action == 0:
            self.pig_x -= PIG_STEP
            self.facing = -1
        elif action == 2:
            self.pig_x += PIG_STEP
            self.facing = 1
        self.moving = action != 1
        self.stride += self.moving
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

        frame = self._pig_frames[(self.stride // 4) % 2 if self.moving else 0]
        if self.facing < 0:
            frame = pygame.transform.flip(frame, True, False)
        surf.blit(frame, (int(self.pig_x) - frame.get_width() // 2, PIG_Y - frame.get_height() // 2))

        pygame.draw.circle(surf, RED, (int(self.stim_x), int(self.stim_y)), STIMULUS_RADIUS)

        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    def close(self):
        pass
