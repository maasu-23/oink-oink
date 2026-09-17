"""Record a GIF of the pig environment under a policy (random by default)."""
import argparse
from pathlib import Path

import imageio
import numpy as np

from oink.env import PigDodgeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/processed/pig_random.gif"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--model", type=Path, default=None, help="optional stable-baselines3 PPO .zip to load")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PigDodgeEnv()
    obs, _ = env.reset(seed=args.seed)

    model = None
    if args.model is not None:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)

    frames = []
    for _ in range(args.steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, _ = env.reset()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=30)
    print(f"Saved {len(frames)}-frame GIF to {args.out}")


if __name__ == "__main__":
    main()
