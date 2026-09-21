"""
Phase 3 — compare connectome-constrained vs. baseline networks: reward curves
across seeds, final-performance summary, and trained-pig GIFs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_monitor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)  # first row is a JSON header comment
    df["cum_steps"] = df["l"].cumsum()
    return df


def smoothed_curve(df: pd.DataFrame, window: int = 20):
    r = df["r"].rolling(window, min_periods=1).mean()
    return df["cum_steps"].to_numpy(), r.to_numpy()


def interpolate_to_grid(x, y, grid):
    return np.interp(grid, x, y, left=np.nan, right=y[-1] if len(y) else np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=Path("data/processed/logs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--variants", nargs="+", default=["connectome", "baseline"])
    parser.add_argument("--threshold", type=float, default=25.0, help="Rolling reward that counts as 'learned'.")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0, 150_000, 300)
    palette = {"connectome": "#4C72B0", "baseline": "#DD8452", "connectome_randinit": "#55A868"}
    labels = {"connectome_randinit": "connectome (random init weights)"}
    summary_rows = []

    plt.figure(figsize=(9, 6))
    for variant in args.variants:
        color = palette.get(variant, "#8172B2")
        curves = []
        for seed in args.seeds:
            path = args.logs_dir / f"{variant}_seed{seed}" / "monitor.monitor.csv"
            df = load_monitor(path)
            x, y = smoothed_curve(df)
            curves.append(interpolate_to_grid(x, y, grid))
            final_reward = df["r"].tail(20).mean()
            crossed = np.nonzero(y >= args.threshold)[0]
            steps_to_threshold = int(x[crossed[0]]) if len(crossed) else None
            summary_rows.append({"variant": variant, "seed": seed, "final_ep_rew_mean": final_reward,
                                 "steps_to_threshold": steps_to_threshold, "n_episodes": len(df)})

        curves = np.array(curves)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        plt.plot(grid, mean_curve, label=labels.get(variant, variant), color=color)
        plt.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (20-episode rolling mean)")
    plt.title("Connectome-constrained vs. baseline network: pig-dodge task")
    plt.legend()
    plt.tight_layout()
    fig_path = args.out_dir / "reward_comparison.png"
    plt.savefig(fig_path, dpi=200)
    print(f"Saved reward comparison plot to {fig_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "results_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print(summary.to_string(index=False))
    print("\nPer-variant final performance (mean +/- std across seeds):")
    print(summary.groupby("variant")["final_ep_rew_mean"].agg(["mean", "std"]).to_string())
    print(f"\nSeeds that reached rolling reward >= {args.threshold:g}:")
    print(summary.groupby("variant")["steps_to_threshold"].agg(lambda c: f"{c.notna().sum()}/{len(c)}").to_string())


if __name__ == "__main__":
    main()
