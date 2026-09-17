"""
Phase 4 — tool implementations the Bedrock supervisor agent can call.

These are plain Python functions with no AWS dependency: the agent layer
(agent.py) is responsible for exposing them to Bedrock's tool-use API and
routing tool-call requests back into these functions. Keeping this module
AWS-free makes the tools independently testable and reusable (e.g. from a
CLI) without a live Bedrock connection.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "data" / "processed" / "logs"
MODELS_DIR = REPO_ROOT / "data" / "processed" / "models"
NODES_CSV = REPO_ROOT / "data" / "processed" / "circuit_nodes.csv"
ADJACENCY_NPZ = REPO_ROOT / "data" / "processed" / "circuit_adjacency.npz"


def list_training_runs() -> dict:
    """List all training runs (variant/seed combinations) with logged metrics."""
    if not LOGS_DIR.exists():
        return {"runs": []}
    runs = []
    for run_dir in sorted(LOGS_DIR.iterdir()):
        monitor_path = run_dir / "monitor.monitor.csv"
        if monitor_path.exists():
            runs.append(run_dir.name)
    return {"runs": runs}


def get_training_metrics(run_name: str, tail_episodes: int = 20) -> dict:
    """Summarize a training run's episode rewards from its monitor.csv log.

    run_name: e.g. "connectome_seed0" (see list_training_runs for valid names).
    """
    monitor_path = LOGS_DIR / run_name / "monitor.monitor.csv"
    if not monitor_path.exists():
        return {"error": f"no monitor log found for run '{run_name}'. Call list_training_runs for valid names."}

    df = pd.read_csv(monitor_path, skiprows=1)
    if df.empty:
        return {"run_name": run_name, "n_episodes": 0}

    tail = df["r"].tail(tail_episodes)
    return {
        "run_name": run_name,
        "n_episodes": len(df),
        "total_timesteps": int(df["l"].sum()),
        "final_reward_mean": round(float(tail.mean()), 3),
        "final_reward_std": round(float(tail.std()), 3),
        "first_10_episodes_mean_reward": round(float(df["r"].head(10).mean()), 3),
        "best_episode_reward": round(float(df["r"].max()), 3),
        "worst_episode_reward": round(float(df["r"].min()), 3),
    }


def compare_variants(variant_a: str = "connectome", variant_b: str = "baseline") -> dict:
    """Compare final performance across all seeds for two network variants."""
    results = {}
    for variant in (variant_a, variant_b):
        runs = [r for r in list_training_runs()["runs"] if r.startswith(f"{variant}_seed")]
        finals = []
        for run in runs:
            m = get_training_metrics(run)
            if "final_reward_mean" in m:
                finals.append(m["final_reward_mean"])
        results[variant] = {
            "seeds_found": len(finals),
            "final_reward_mean_across_seeds": round(float(np.mean(finals)), 3) if finals else None,
            "final_reward_std_across_seeds": round(float(np.std(finals)), 3) if finals else None,
            "per_seed_final_rewards": finals,
        }
    return results


def get_circuit_summary() -> dict:
    """Summarize the extracted connectome circuit: neuron counts and edge counts per layer."""
    if not NODES_CSV.exists():
        return {"error": "circuit_nodes.csv not found — has Phase 1 extraction been run?"}
    nodes = pd.read_csv(NODES_CSV)
    adj = sp.load_npz(ADJACENCY_NPZ).tocsr()
    idx = {role: nodes.loc[nodes["role"] == role, "node_index"].to_numpy() for role in ("input", "processing", "output")}

    def edge_count(pre, post):
        return int(adj[np.ix_(pre, post)].nnz)

    return {
        "n_input_neurons": len(idx["input"]),
        "n_processing_neurons": len(idx["processing"]),
        "n_output_neurons": len(idx["output"]),
        "input_to_input_edges": edge_count(idx["input"], idx["input"]),
        "input_to_processing_edges": edge_count(idx["input"], idx["processing"]),
        "processing_to_output_edges": edge_count(idx["processing"], idx["output"]),
        "biology": (
            "input = T4/T5 motion-detecting neurons, processing = HS/VS lobula "
            "plate tangential cells, output = descending neurons (DN) carrying "
            "motor commands. Extracted from the FlyWire FAFB v783 connectome."
        ),
    }


def list_neurons(role: str, limit: int = 10) -> dict:
    """List sample FlyWire root_ids for a given neuron role.

    role: "input" (T4/T5), "processing" (HS/VS), or "output" (descending neurons).
    Use this to discover real root_ids before calling get_neuron_connections.
    """
    if role not in ("input", "processing", "output"):
        return {"error": "role must be 'input', 'processing', or 'output'"}
    nodes = pd.read_csv(NODES_CSV)
    ids = nodes.loc[nodes["role"] == role, "root_id"].head(limit).tolist()
    return {"role": role, "root_ids": [int(i) for i in ids], "total_in_role": int((nodes["role"] == role).sum())}


def get_neuron_connections(root_id: int, direction: str = "both", limit: int = 20) -> dict:
    """Look up a specific neuron's synaptic partners by its FlyWire root_id.

    direction: "upstream" (presynaptic partners), "downstream" (postsynaptic
    partners), or "both".
    """
    nodes = pd.read_csv(NODES_CSV)
    if root_id not in nodes["root_id"].values:
        return {"error": f"root_id {root_id} not found in extracted circuit nodes."}

    adj = sp.load_npz(ADJACENCY_NPZ).tocsr()
    node_row = nodes.loc[nodes["root_id"] == root_id].iloc[0]
    node_index = int(node_row["node_index"])
    result = {"root_id": root_id, "role": node_row["role"]}

    id_by_index = nodes.set_index("node_index")["root_id"]
    role_by_index = nodes.set_index("node_index")["role"]

    if direction in ("downstream", "both"):
        row = adj.getrow(node_index).tocoo()
        partners = sorted(zip(row.col, row.data), key=lambda p: -abs(p[1]))[:limit]
        result["downstream"] = [
            {"root_id": int(id_by_index[c]), "role": role_by_index[c], "weight": float(w)} for c, w in partners
        ]

    if direction in ("upstream", "both"):
        col = adj.getcol(node_index).tocoo()
        partners = sorted(zip(col.row, col.data), key=lambda p: -abs(p[1]))[:limit]
        result["upstream"] = [
            {"root_id": int(id_by_index[r]), "role": role_by_index[r], "weight": float(w)} for r, w in partners
        ]

    return result


def trigger_retrain(variant: str, seed: int, timesteps: int = 150_000) -> dict:
    """Kick off a new PPO training run in the background and return immediately.

    variant: "connectome" or "baseline". Overwrites any existing model/log
    for this variant+seed. Use sparingly — each run takes real wall-clock
    time and CPU.
    """
    if variant not in ("connectome", "baseline"):
        return {"error": "variant must be 'connectome' or 'baseline'"}

    log_dir = LOGS_DIR / f"{variant}_seed{seed}"
    out_path = MODELS_DIR / f"ppo_{variant}_seed{seed}.zip"
    stdout_path = log_dir / "retrain_stdout.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "oink.train",
        "--variant", variant,
        "--seed", str(seed),
        "--timesteps", str(timesteps),
        "--out", str(out_path),
        "--log-dir", str(log_dir),
    ]
    with open(stdout_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

    return {
        "status": "started",
        "pid": proc.pid,
        "variant": variant,
        "seed": seed,
        "timesteps": timesteps,
        "stdout_log": str(stdout_path),
        "note": "Runs in the background; poll get_training_metrics once complete.",
    }


TOOL_REGISTRY = {
    "list_training_runs": list_training_runs,
    "get_training_metrics": get_training_metrics,
    "compare_variants": compare_variants,
    "get_circuit_summary": get_circuit_summary,
    "list_neurons": list_neurons,
    "get_neuron_connections": get_neuron_connections,
    "trigger_retrain": trigger_retrain,
}
