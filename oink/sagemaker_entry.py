"""
SageMaker training-job entry point: adapts SageMaker's conventions to oink.train.

SageMaker runs this inside a PyTorch container with:
  SM_CHANNEL_CIRCUIT  -> directory holding circuit_nodes.csv + circuit_adjacency.npz
  SM_MODEL_DIR        -> whatever is written here is tarred to S3 as model.tar.gz
and hyperparameters arrive as CLI flags (--variant, --seed, --timesteps).

We write both the trained policy and the Monitor CSV into SM_MODEL_DIR so a
single tarball per job carries everything compare_results.py needs.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # SageMaker runs `python oink/sagemaker_entry.py`

from oink.train import make_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--checkpoint-every", type=int, default=0, help="Also save the policy every N steps (0 = off).")
    p.add_argument("--circuit-dir", type=Path, default=Path(os.environ.get("SM_CHANNEL_CIRCUIT", "data/processed")))
    p.add_argument("--model-dir", type=Path, default=Path(os.environ.get("SM_MODEL_DIR", "data/processed/sagemaker_out")))
    args = p.parse_args()

    run_name = f"{args.variant}_seed{args.seed}"
    log_dir = args.model_dir / "logs" / run_name
    model, _ = make_model(
        args.variant,
        args.circuit_dir / "circuit_nodes.csv",
        args.circuit_dir / "circuit_adjacency.npz",
        args.seed,
        log_dir,
    )
    callback = None
    if args.checkpoint_every:
        from stable_baselines3.common.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=args.checkpoint_every, save_path=str(args.model_dir / "checkpoints"), name_prefix=f"ppo_{run_name}"
        )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    out = args.model_dir / "models" / f"ppo_{run_name}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
