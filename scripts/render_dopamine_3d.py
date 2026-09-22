"""
Demo video — the real fly circuit lighting up in 3D while the pig plays.

Left panel: the pig dodging dripstones with a trained connectome model.
Right panel: the FlyWire skeletons of our T4/T5 -> HS/VS -> DN circuit,
each neuron coloured by its *actual* activation in the network that frame,
plus the fly's real reward-dopamine neurons (the PAM cluster, which feeds the
mushroom body) glowing gold on a "dopamine hit".

What drives the dopamine glow (--glow):
  rpe     the network's reward prediction error, r_t + gamma*V(s_t+1) - V(s_t),
          read from PPO's critic. This is what dopamine neurons encode in real
          brains (Schultz 1997), so it is the honest choice -- but a
          well-trained critic is rarely surprised, so the flashes are small.
  reward  the raw reward event (+1 dodge / -10 hit). Louder, less faithful.

The PAM neurons are NOT part of the trained network; they stand in for PPO's
critic so the reward signal has a real anatomical home in the picture. Say
so in any caption.

No changes to the network are needed: activations are captured with a forward
hook on the trunk's shared Tanh, which fires once per layer (input,
processing, output) per forward pass.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import navis
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))
from render_3d_circuit import extract_skeletons  # noqa: E402

from oink.env import PigDodgeEnv  # noqa: E402

ROLE_BASE = {"input": (0.30, 0.50, 0.90), "processing": (0.95, 0.60, 0.10), "output": (0.85, 0.15, 0.25)}
ROLE_LABEL = {"input": "T4/T5 motion detectors", "processing": "HS/VS tangential cells", "output": "descending neurons (to muscles)"}
DA_DIM = np.array([0.42, 0.42, 0.48])
DA_GOLD = np.array([1.00, 0.82, 0.15])
DA_BLUE = np.array([0.25, 0.40, 0.95])
BG = "#06060a"


# ----------------------------------------------------------------------------
# 1. Which neurons to draw
# ----------------------------------------------------------------------------
def pick_neurons(nodes_csv: Path, cell_types_csv: Path, max_per_role: int, max_pam: int) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_csv)
    # ConnectomeNetwork indexes each layer's activation vector by CSV order
    # within the role, so remember each neuron's position before subsampling.
    nodes["pos_in_role"] = nodes.groupby("role").cumcount()
    sub = pd.concat([g.sample(min(len(g), max_per_role), random_state=0) for _, g in nodes.groupby("role")])

    ct = pd.read_csv(cell_types_csv)
    pam = ct[ct["primary_type"].astype(str).str.startswith("PAM")].copy()
    pam = pam.sample(min(len(pam), max_pam), random_state=0)
    pam = pd.DataFrame({"root_id": pam["root_id"], "role": "dopamine", "pos_in_role": -1})
    return pd.concat([sub[["root_id", "role", "pos_in_role"]], pam], ignore_index=True)


def load_segments(paths: list[Path], downsample: int):
    """Returns (segments [N,2,3], owner [N]) where owner is the index of the path each segment belongs to."""
    neurons = navis.read_swc(paths, parallel=False)
    by_id = {int(n.id): n for n in neurons}
    segs, owner = [], []
    for i, p in enumerate(paths):
        n = by_id.get(int(p.stem))
        if n is None:
            continue
        if downsample > 1:
            n = navis.downsample_neuron(n, downsample, inplace=False)
        nd = n.nodes.set_index("node_id")
        child = nd[nd["parent_id"] != -1]
        a = child[["x", "y", "z"]].to_numpy()
        b = nd.loc[child["parent_id"], ["x", "y", "z"]].to_numpy()
        seg = np.stack([a, b], axis=1)
        # FlyWire: x lateral, y dorsal->ventral, z anterior->posterior. Matplotlib
        # draws z as "up", so plot (x, z, -y) to put the dorsal side on top.
        seg = np.stack([seg[..., 0], seg[..., 2], -seg[..., 1]], axis=-1)
        segs.append(seg)
        owner.append(np.full(len(a), i))
    return np.concatenate(segs), np.concatenate(owner)


# ----------------------------------------------------------------------------
# 2. Play one episode and record everything
# ----------------------------------------------------------------------------
def record_episode(model_path: Path, env_seed: int, max_steps: int):
    from oink.policy import load_ppo

    model = load_ppo(model_path, device="cpu")
    policy = model.policy
    layers: list[np.ndarray] = []
    hook = policy.features_extractor.trunk.act.register_forward_hook(lambda m, i, o: layers.append(o.detach().numpy()[0]))

    env = PigDodgeEnv()
    obs, _ = env.reset(seed=env_seed)
    states, acts, values, rewards = [], {"input": [], "processing": [], "output": []}, [], []
    terminated = False
    for _ in range(max_steps):
        layers.clear()
        with torch.no_grad():
            action, value, _ = policy(torch.as_tensor(obs[None], dtype=torch.float32), deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        for k, v in zip(("input", "processing", "output"), layers):
            acts[k].append(v)
        values.append(float(value))
        rewards.append(float(reward))
        states.append(game_state(env))
        if terminated or truncated:
            break
    hook.remove()

    values = np.array(values)
    rewards = np.array(rewards)
    next_values = np.append(values[1:], 0.0 if terminated else values[-1])
    rpe = rewards + model.gamma * next_values - values
    return states, {k: np.stack(v) for k, v in acts.items()}, rewards, values, rpe


GAME_KEYS = ("pig_x", "stim_x", "stim_y", "facing", "moving", "stride")


def game_state(env: PigDodgeEnv) -> dict:
    return {k: getattr(env, k) for k in GAME_KEYS}


def render_game(env: PigDodgeEnv, a: dict, b: dict | None, frac: float) -> np.ndarray:
    """Draw the game between two recorded steps (render-only: the physics already happened).

    The dripstone is not interpolated across a respawn (it jumps from the ground
    back to the sky), so on those steps it simply holds its position.
    """
    for k in GAME_KEYS:
        setattr(env, k, a[k])
    if b is not None and frac > 0:
        env.pig_x = a["pig_x"] + (b["pig_x"] - a["pig_x"]) * frac
        if b["stim_y"] >= a["stim_y"]:
            env.stim_x = a["stim_x"] + (b["stim_x"] - a["stim_x"]) * frac
            env.stim_y = a["stim_y"] + (b["stim_y"] - a["stim_y"]) * frac
    return env.render()


def glow_trace(rewards: np.ndarray, rpe: np.ndarray, mode: str, scale: float, decay: float) -> np.ndarray:
    """Signed dopamine glow in [-1, 1] per frame, with an exponential release so hits linger."""
    if mode == "rpe":
        drive = np.clip(rpe / scale, -1, 1)
    else:
        drive = np.where(rewards >= 1.0, 1.0, np.where(rewards <= -1.0, -1.0, 0.0))
    glow = np.zeros_like(drive)
    g = 0.0
    for t, d in enumerate(drive):
        g = d if abs(d) > abs(g * decay) else g * decay
        glow[t] = g
    return glow


# ----------------------------------------------------------------------------
# 3. Render
# ----------------------------------------------------------------------------
def role_colors(base, act: np.ndarray) -> np.ndarray:
    """Per-neuron RGBA: quiet = dim/translucent, active = bright and whiter."""
    a = np.abs(act)[:, None]
    rgb = np.array(base)[None, :]
    rgb = rgb + (1.0 - rgb) * 0.55 * a
    alpha = 0.10 + 0.90 * a
    return np.concatenate([rgb, alpha], axis=1)


def dopamine_colors(glow: float, n: int) -> np.ndarray:
    if glow >= 0:
        rgb = DA_DIM + (DA_GOLD - DA_DIM) * glow
        alpha = 0.22 + 0.78 * glow
    else:
        rgb = DA_DIM + (DA_BLUE - DA_DIM) * (-glow)
        alpha = 0.22 + 0.60 * (-glow)
    return np.tile(np.append(rgb, alpha), (n, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, nargs="+", default=[REPO_ROOT / "data/processed/models/ppo_connectome_seed0.zip"],
                   help="One or more checkpoints; each gets its own segment, played back to back.")
    p.add_argument("--label", type=str, nargs="+", default=None, help="Banner text per segment (same count as --model).")
    p.add_argument("--env-seed", type=int, nargs="+", default=[42], help="Env seed per segment (one value = shared).")
    p.add_argument("--max-steps", type=int, nargs="+", default=[500], help="Step cap per segment (one value = shared).")
    p.add_argument("--skeleton-zip", type=Path, default=Path.home() / "Downloads" / "sk_lod1_783_healed.zip")
    p.add_argument("--skeleton-dir", type=Path, default=REPO_ROOT / "data/raw/skeletons")
    p.add_argument("--nodes-csv", type=Path, default=REPO_ROOT / "data/processed/circuit_nodes.csv")
    p.add_argument("--cell-types-csv", type=Path, default=Path.home() / "Downloads" / "consolidated_cell_types.csv.gz")
    p.add_argument("--max-per-role", type=int, default=300)
    p.add_argument("--max-pam", type=int, default=120)
    p.add_argument("--downsample", type=int, default=8, help="Skeleton node downsampling factor (speed).")
    p.add_argument("--glow", choices=["rpe", "reward"], default="rpe")
    p.add_argument("--rpe-scale", type=float, default=0.3, help="RPE that saturates the glow.")
    p.add_argument("--decay", type=float, default=0.85)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--substeps", type=int, default=2, help="Video frames per game step (positions/colours interpolated).")
    p.add_argument("--every", type=int, default=1, help="Render every Nth video frame (preview: keeps timing, lowers fps).")
    p.add_argument("--hold", type=int, default=30, help="Extra frames to linger on a hit / segment end.")
    p.add_argument("--fade", type=int, default=15, help="Frames for each fade in/out at segment boundaries.")
    p.add_argument("--zoom", type=float, default=0.62, help="Fraction of the data's max range shown; smaller = closer.")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/pigbrain_dopamine.mp4")
    p.add_argument("--subtitle", default="a real fruit-fly circuit (FlyWire), trained with PPO on Amazon SageMaker, live in 3D")
    p.add_argument("--test-frame", type=int, default=None, help="Save a single PNG of this step instead of a video.")
    args = p.parse_args()

    # --- neurons --------------------------------------------------------------
    sel = pick_neurons(args.nodes_csv, args.cell_types_csv, args.max_per_role, args.max_pam)
    print(f"Extracting {len(sel)} skeletons...")
    paths = extract_skeletons(args.skeleton_zip, sel["root_id"].tolist(), args.skeleton_dir)
    sel = sel.set_index("root_id").loc[[int(pth.stem) for pth in paths]].reset_index()
    groups = {}
    for role, g in sel.groupby("role", sort=False):
        gpaths = [args.skeleton_dir / f"{rid}.swc" for rid in g["root_id"]]
        segs, owner = load_segments(gpaths, args.downsample)
        groups[role] = dict(segs=segs, owner=owner, pos=g["pos_in_role"].to_numpy(), n=len(g))
        print(f"  {role:<10} {len(g):>4} neurons  {len(segs):>7} segments")

    # --- episodes (one per segment) -------------------------------------------
    n_seg = len(args.model)
    labels = args.label or [m.stem for m in args.model]
    seeds = args.env_seed * n_seg if len(args.env_seed) == 1 else args.env_seed
    caps = args.max_steps * n_seg if len(args.max_steps) == 1 else args.max_steps
    if not (len(labels) == len(seeds) == len(caps) == n_seg):
        raise SystemExit("--model, --label, --env-seed and --max-steps must have matching counts")

    states, rewards, values, rpe, seg_of, step_in_seg = [], [], [], [], [], []
    acts = {"input": [], "processing": [], "output": []}
    for s, (model_path, seed, cap) in enumerate(zip(args.model, seeds, caps)):
        print(f"Segment {s + 1}/{n_seg}: {model_path.name} (env seed {seed}, cap {cap})...")
        f, a, r, v, d = record_episode(model_path, seed, cap)
        states += f
        for k in acts:
            acts[k].append(a[k])
        rewards.append(r)
        values.append(v)
        rpe.append(d)
        seg_of += [s] * len(f)
        step_in_seg += list(range(len(f)))
        print(f"  {len(f)} steps, {(r >= 1).sum()} dodges, hit={r[-1] <= -1}, "
              f"RPE on dodges: {np.round(d[r >= 1.0], 2).tolist()}")
    acts = {k: np.concatenate(v) for k, v in acts.items()}
    rewards, values, rpe = map(np.concatenate, (rewards, values, rpe))
    seg_of, step_in_seg = np.array(seg_of), np.array(step_in_seg)
    n_steps = len(states)
    game_env = PigDodgeEnv()
    # Glow and running totals restart at each segment boundary.
    glow = np.concatenate([glow_trace(rewards[seg_of == s], rpe[seg_of == s], args.glow, args.rpe_scale, args.decay) for s in range(n_seg)])
    dodges = np.concatenate([np.cumsum(rewards[seg_of == s] >= 1.0) for s in range(n_seg)])
    total_reward = np.concatenate([np.cumsum(rewards[seg_of == s]) for s in range(n_seg)])

    # --- figure ---------------------------------------------------------------
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax_pig = fig.add_axes([0.035, 0.20, 0.40, 0.62])
    ax_pig.set_axis_off()
    ax_pig.set_facecolor(BG)
    first = render_game(game_env, states[0], None, 0.0)
    img = ax_pig.imshow(first, interpolation="nearest")
    ax_pig.set_xlim(0, first.shape[1])
    ax_pig.set_ylim(first.shape[0], 0)

    ax = fig.add_axes([0.40, 0.02, 0.60, 0.92], projection="3d", facecolor=BG)
    ax.set_axis_off()
    collections = {}
    for role, g in groups.items():
        lc = Line3DCollection(g["segs"], linewidths=0.5)
        ax.add_collection3d(lc)
        collections[role] = lc

    all_pts = np.concatenate([g["segs"].reshape(-1, 3) for g in groups.values()])
    lo, hi = all_pts.min(0), all_pts.max(0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * args.zoom
    ax.set_xlim3d(center[0] - half, center[0] + half)
    ax.set_ylim3d(center[1] - half, center[1] + half)
    ax.set_zlim3d(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))

    # --- text -----------------------------------------------------------------
    fig.text(0.035, 0.93, "PigBrain", color="white", fontsize=26, fontweight="bold", va="center")
    fig.text(0.035, 0.885, args.subtitle,
             color="#bbbbcc", fontsize=12, va="center")
    banner = fig.text(0.235, 0.845, "", color="white", fontsize=15, fontweight="bold", ha="center", va="center")
    status = fig.text(0.035, 0.155, "", color="#dddddd", fontsize=12, family="monospace", va="center")
    flash = fig.text(0.235, 0.52, "", color=DA_GOLD, fontsize=30, fontweight="bold", ha="center", va="center")

    ly = 0.16
    for role in ("input", "processing", "output"):
        fig.text(0.62, ly, "█", color=ROLE_BASE[role], fontsize=11, va="center")
        fig.text(0.64, ly, ROLE_LABEL[role] + "  (colour = live activation)", color="#cccccc", fontsize=10, va="center")
        ly -= 0.035
    fig.text(0.62, ly, "█", color=DA_GOLD, fontsize=11, va="center")
    da_label = "PAM dopamine neurons  (glow = reward prediction error)" if args.glow == "rpe" else \
               "PAM dopamine neurons  (glow = reward: +1 dodge / -10 hit)"
    fig.text(0.64, ly, da_label, color="#cccccc", fontsize=10, va="center")

    # reward-prediction-error gauge under the pig
    gauge_ax = fig.add_axes([0.035, 0.085, 0.40, 0.035])
    gauge_ax.set_xlim(-1, 1)
    gauge_ax.set_ylim(0, 1)
    gauge_ax.set_axis_off()
    gauge_ax.axvline(0, color="#555566", lw=1)
    gauge_bar = gauge_ax.barh(0.5, 0, height=0.8, color=DA_GOLD)[0]
    fig.text(0.235, 0.05, "dopamine signal  =  reward prediction error  (\"better / worse than I expected\")",
             color="#9999aa", fontsize=9, ha="center", va="center")

    # --- timeline: (step, frac) per video frame, with holds and fades ---------
    # Each game step gets `substeps` video frames; frac in [0,1) interpolates
    # toward the next step of the same segment. Hits and segment ends linger,
    # and segment boundaries fade to black and back. A flash event is the
    # frame index at which a reward landed, so the text can ease in/out.
    timeline: list[tuple[int, float]] = []
    for t in range(n_steps):
        last_in_seg = t + 1 == n_steps or seg_of[t + 1] != seg_of[t]
        for sub in range(args.substeps):
            timeline.append((t, 0.0 if last_in_seg else sub / args.substeps))
        if last_in_seg:
            timeline += [(t, 0.0)] * args.hold
    total_frames = len(timeline)
    seg_starts = [i for i in range(total_frames) if i == 0 or seg_of[timeline[i][0]] != seg_of[timeline[i - 1][0]]]
    fade = args.fade

    def frame_alpha(i: int) -> float:
        a = 1.0
        for s in seg_starts:
            if s <= i < s + fade:  # fade in
                a = min(a, (i - s + 1) / fade)
            if s - fade <= i < s:  # fade out before the next segment
                a = min(a, (s - i) / fade)
        if i >= total_frames - fade:
            a = min(a, (total_frames - i) / fade)
        return a

    events = np.where(np.abs(rewards) >= 1.0)[0]
    step_first_frame = {}
    for i, (t, frac) in enumerate(timeline):
        step_first_frame.setdefault(t, i)

    def draw(i: int):
        t, frac = timeline[i]
        nxt = t + 1 if (t + 1 < n_steps and seg_of[t + 1] == seg_of[t]) else None
        img.set_data(render_game(game_env, states[t], states[nxt] if nxt is not None else None, frac))

        def lerp(arr):
            return arr[t] if nxt is None else arr[t] + (arr[nxt] - arr[t]) * frac

        for role in ("input", "processing", "output"):
            g = groups[role]
            per_neuron = role_colors(ROLE_BASE[role], lerp(acts[role])[g["pos"]])
            collections[role].set_color(per_neuron[g["owner"]])
        gl = float(lerp(glow))
        collections["dopamine"].set_color(dopamine_colors(gl, len(groups["dopamine"]["owner"])))
        ax.view_init(elev=18, azim=-60 + 360.0 * i / total_frames)

        banner.set_text(labels[seg_of[t]])
        status.set_text(f"step {step_in_seg[t] + 1:>3}   dodges {dodges[t]:>2}   reward {total_reward[t]:>6.2f}   expects {values[t]:5.2f}")
        gauge_bar.set_width(gl)
        gauge_bar.set_x(min(0, gl))
        gauge_bar.set_color(DA_GOLD if gl >= 0 else DA_BLUE)

        # Flash text: ease in over a few frames, hold, ease out.
        recent = [e for e in events if seg_of[e] == seg_of[t] and step_first_frame[e] <= i]
        if recent:
            e = recent[-1]
            age = i - step_first_frame[e]
            rise, hold_n, fall = 6, 18, 20
            if age < rise:
                a = age / rise
            elif age < rise + hold_n:
                a = 1.0
            else:
                a = max(0.0, 1 - (age - rise - hold_n) / fall)
            flash.set_text("+1  dodged!" if rewards[e] >= 1 else "-10  ouch")
            flash.set_color(DA_GOLD if rewards[e] >= 1 else DA_BLUE)
            flash.set_alpha(a)
        else:
            flash.set_alpha(0.0)

    def grab() -> np.ndarray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    if args.test_frame is not None:
        draw(step_first_frame[args.test_frame])
        out = args.out.with_suffix(".png")
        fig.savefig(out, dpi=100, facecolor=BG)
        print(f"Saved test frame to {out}")
        return

    import imageio_ffmpeg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = list(range(0, total_frames, args.every))
    fps = args.fps / args.every
    writer = imageio_ffmpeg.write_frames(
        str(args.out), (1280, 720), fps=fps, codec="libx264", bitrate="6000k", pix_fmt_out="yuv420p", macro_block_size=1,
    )
    writer.send(None)
    for n, i in enumerate(frame_ids):
        draw(i)
        rgb = grab()
        a = frame_alpha(i)
        if a < 1.0:
            rgb = (rgb.astype(np.float32) * a).astype(np.uint8)
        writer.send(np.ascontiguousarray(rgb))
        if n % 50 == 0:
            print(f"  frame {n + 1}/{len(frame_ids)}", flush=True)
    writer.close()
    print(f"Saved video to {args.out}  ({len(frame_ids)} frames @ {fps:g} fps = {len(frame_ids) / fps:.1f}s)")


if __name__ == "__main__":
    main()
