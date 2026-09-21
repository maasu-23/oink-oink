"""
Phase 2/3 — train the connectome-constrained network and the baseline on the
pig-dodge task with PPO, for comparison.
"""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from oink.env import PigDodgeEnv
from oink.policy import CircuitFeaturesExtractor


def make_model(variant: str, nodes_csv: Path, adjacency_npz: Path, seed: int, log_dir: Path | None = None):
    monitor_path = str(log_dir / "monitor") if log_dir is not None else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: Monitor(PigDodgeEnv(), filename=monitor_path), n_envs=1, seed=seed)
    policy_kwargs = dict(
        features_extractor_class=CircuitFeaturesExtractor,
        features_extractor_kwargs=dict(nodes_csv=nodes_csv, adjacency_npz=adjacency_npz, variant=variant, seed=seed),
        net_arch=[],  # circuit trunk's output feeds SB3's action/value heads directly
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, seed=seed, n_steps=512, batch_size=64)
    return model, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["connectome", "connectome_randinit", "baseline"], required=True)
    parser.add_argument("--nodes-csv", type=Path, default=Path("data/processed/circuit_nodes.csv"))
    parser.add_argument("--adjacency-npz", type=Path, default=Path("data/processed/circuit_adjacency.npz"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or Path(f"data/processed/ppo_{args.variant}_seed{args.seed}.zip")
    log_dir = args.log_dir or Path(f"data/processed/logs/{args.variant}_seed{args.seed}")
    model, env = make_model(args.variant, args.nodes_csv, args.adjacency_npz, args.seed, log_dir)
    model.learn(total_timesteps=args.timesteps)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"Saved trained model to {out}")


if __name__ == "__main__":
    main()
