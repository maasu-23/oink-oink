"""
Bonus deliverable — fancy 3D rendering of the extracted optomotor circuit.

Loads the real FlyWire skeleton (.swc) meshes for every neuron in our
circuit (input=T4/T5, processing=HS/VS, output=DN), colors them by role,
and renders a rotating GIF for the README.

Skeletons are extracted on demand from the large FlyWire skeleton archive
(not committed to the repo -- see --skeleton-zip) into data/raw/skeletons/,
which IS gitignored (data/raw/ is raw/derived-from-external-source data).
"""
import argparse
import zipfile
from pathlib import Path

import navis
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_COLORS = {"input": (0.3, 0.5, 0.9), "processing": (0.95, 0.6, 0.1), "output": (0.85, 0.15, 0.25)}


def extract_skeletons(skeleton_zip: Path, root_ids, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [f"{rid}.swc" for rid in root_ids]
    missing = [w for w in wanted if not (out_dir / w).exists()]
    if missing:
        with zipfile.ZipFile(skeleton_zip) as z:
            names = set(z.namelist())
            to_extract = [w for w in missing if w in names]
            z.extractall(out_dir, members=to_extract)
    return [out_dir / w for w in wanted if (out_dir / w).exists()]


def load_neurons(paths: list[Path], nodes: pd.DataFrame) -> navis.NeuronList:
    role_by_id = dict(zip(nodes["root_id"], nodes["role"]))
    neurons = navis.read_swc(paths, parallel=False)
    for n in neurons:
        n.role = role_by_id.get(int(n.id), "unknown")
    return neurons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton-zip", type=Path, default=Path.home() / "Downloads" / "sk_lod1_783_healed.zip")
    parser.add_argument("--nodes-csv", type=Path, default=REPO_ROOT / "data/processed/circuit_nodes.csv")
    parser.add_argument("--skeleton-dir", type=Path, default=REPO_ROOT / "data/raw/skeletons")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/circuit_3d.gif")
    parser.add_argument("--max-per-role", type=int, default=None, help="Subsample per role for a faster/lighter render.")
    parser.add_argument("--n-frames", type=int, default=60)
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes_csv)
    if args.max_per_role:
        nodes = pd.concat(
            [g.sample(min(len(g), args.max_per_role), random_state=0) for _, g in nodes.groupby("role")]
        )

    print(f"Extracting {len(nodes)} skeletons...")
    paths = extract_skeletons(args.skeleton_zip, nodes["root_id"].tolist(), args.skeleton_dir)
    print(f"Loading {len(paths)} skeletons into navis...")
    neurons = load_neurons(paths, nodes)

    colors = {n.id: ROLE_COLORS.get(n.role, (0.5, 0.5, 0.5)) for n in neurons}

    print("Rendering rotating 3D view...")
    fig, ax = navis.plot2d(
        neurons,
        color=colors,
        method="3d",
        linewidth=0.5,
        soma=False,
        figsize=(8, 8),
    )
    ax.set_axis_off()

    # Equal-range cube around the data so rotation doesn't distort proportions,
    # and pull the camera in tight since autoscale leaves a lot of dead space.
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    centers = [(lo + hi) / 2 for lo, hi in (xlim, ylim, zlim)]
    half_range = max(hi - lo for lo, hi in (xlim, ylim, zlim)) / 2 * 0.55
    ax.set_xlim3d(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim3d(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim3d(centers[2] - half_range, centers[2] + half_range)
    ax.set_box_aspect((1, 1, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)

    ax.set_title(
        "PigBrain optomotor circuit -- blue=T4/T5 input  orange=HS/VS processing  red=DN output",
        fontsize=11,
    )

    import matplotlib.animation as animation

    def rotate(angle):
        ax.view_init(elev=15, azim=angle)
        return ax

    angles = np.linspace(0, 360, args.n_frames, endpoint=False)
    anim = animation.FuncAnimation(fig, rotate, frames=angles, interval=50)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer="pillow", fps=20)
    print(f"Saved rotating 3D GIF to {args.out}")


if __name__ == "__main__":
    main()
