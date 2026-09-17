#!/usr/bin/env python3
"""
Phase 1 — Extract the FlyWire optomotor circuit topology.

Circuit boundary (FAFB v783, female adult fly brain):
  input      = T4 Neuron, T5 Neuron   (motion detectors, optic lobe)
  processing = HS, VS                 (lobula plate tangential cells)
  output     = DN                     (descending neurons -> motor output)

Reads FlyWire Codex CSV exports (neurons.csv.gz, visual_neuron_types.csv.gz,
connections_princeton.csv.gz), builds a signed, weighted adjacency matrix
over the induced subgraph on those neurons, and writes it out as a sparse
.npz plus a node-metadata table and a subgraph visualization.

Usage:
    python scripts/extract_circuit.py \
        --data-dir ~/Downloads \
        --out-dir data/processed
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

INPUT_FAMILIES = ["T4 Neuron", "T5 Neuron"]
PROCESSING_FAMILIES = ["HS", "VS"]

# NOTE: visual_neuron_types.csv.gz has its own family=="DN" tag, but those 20
# neurons turn out to have ZERO synapses from HS/VS in this dataset -- they
# are an unrelated sample of the ~350 known descending-neuron types. The
# actual descending partners of HS/VS carry short type-code labels (DNp_L,
# DNg_R, DNa_L, ...) in labels.csv.gz instead. We detect the real output
# layer by looking at who HS/VS and T4/T5 actually synapse onto and matching
# that naming convention, rather than trusting the unrelated family tag.
DN_LABEL_PATTERN = re.compile(r"^DN[A-Za-z]{0,4}\d{0,3}(_[LR])?$")

# Predicted-neurotransmitter -> synapse sign. ACh is excitatory; GABA and
# glutamate are treated as inhibitory (glutamate is inhibitory at fly NMJs
# via GluCl and is commonly treated as inhibitory in central circuits too).
# Everything else (dopamine, serotonin, octopamine, unknown) is neuromodulatory
# / ambiguous and is left unsigned (0) rather than guessed.
NT_SIGN = {
    "ACH": 1,
    "GABA": -1,
    "GLUT": -1,
}


def load_tables(data_dir: Path):
    neurons = pd.read_csv(data_dir / "neurons.csv.gz")
    visual_types = pd.read_csv(data_dir / "visual_neuron_types.csv.gz")
    connections = pd.read_csv(data_dir / "connections_princeton.csv.gz")
    labels = pd.read_csv(data_dir / "labels.csv.gz")
    return neurons, visual_types, connections, labels


def find_descending_outputs(upstream_ids: set, connections: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Neurons downstream of `upstream_ids` whose label matches DN naming (DNp_L, DNg_R, ...)."""
    targets = set(connections.loc[connections["pre_root_id"].isin(upstream_ids), "post_root_id"])
    lab = labels[labels["root_id"].isin(targets)].copy()
    lab["prefix"] = lab["label"].fillna("").str.split(";").str[0].str.strip()
    lab["is_dn"] = lab["prefix"].apply(lambda s: bool(DN_LABEL_PATTERN.match(s)))
    dn = lab.loc[lab["is_dn"], ["root_id", "prefix"]].drop_duplicates(subset="root_id")
    return dn.rename(columns={"prefix": "type"})


def build_node_table(visual_types: pd.DataFrame, connections: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    in_proc_families = INPUT_FAMILIES + PROCESSING_FAMILIES
    nodes = visual_types[visual_types["family"].isin(in_proc_families)].copy()
    nodes = nodes.drop_duplicates(subset="root_id")
    nodes["role"] = np.where(nodes["family"].isin(INPUT_FAMILIES), "input", "processing")

    upstream_ids = set(nodes["root_id"])
    dn = find_descending_outputs(upstream_ids, connections, labels)
    dn["family"] = "DN"
    dn["role"] = "output"
    dn["side"] = np.nan

    nodes = pd.concat([nodes[["root_id", "type", "family", "role", "side"]], dn[["root_id", "type", "family", "role", "side"]]])
    nodes = nodes.drop_duplicates(subset="root_id").reset_index(drop=True)
    nodes["node_index"] = nodes.index
    return nodes[["node_index", "root_id", "type", "family", "role", "side"]]


def build_adjacency(nodes: pd.DataFrame, neurons: pd.DataFrame, connections: pd.DataFrame):
    node_ids = set(nodes["root_id"])
    id_to_index = dict(zip(nodes["root_id"], nodes["node_index"]))

    sub = connections[
        connections["pre_root_id"].isin(node_ids) & connections["post_root_id"].isin(node_ids)
    ].copy()

    # Sign each edge from the presynaptic neuron's predicted neurotransmitter.
    nt_lookup = neurons.set_index("root_id")["nt_type"]
    sub["nt_type"] = sub["pre_root_id"].map(nt_lookup).fillna(sub["nt_type"])
    sub["sign"] = sub["nt_type"].map(NT_SIGN).fillna(0)

    # Aggregate multiple synapses/neuropils between the same pre/post pair.
    agg = (
        sub.groupby(["pre_root_id", "post_root_id"])
        .agg(syn_count=("syn_count", "sum"), sign=("sign", "first"))
        .reset_index()
    )

    rows = agg["pre_root_id"].map(id_to_index).to_numpy()
    cols = agg["post_root_id"].map(id_to_index).to_numpy()
    weights = (agg["syn_count"] * agg["sign"]).to_numpy(dtype=np.float32)

    n = len(nodes)
    adj = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    return adj, agg


def sanity_check(nodes: pd.DataFrame, adj: sp.csr_matrix):
    counts = nodes["role"].value_counts()
    print("Node counts by role:")
    print(counts.to_string())
    print(f"\nTotal neurons: {len(nodes)}")
    print(f"Total signed edges (nonzero entries): {adj.nnz}")

    input_idx = nodes.loc[nodes["role"] == "input", "node_index"].to_numpy()
    proc_idx = nodes.loc[nodes["role"] == "processing", "node_index"].to_numpy()
    output_idx = nodes.loc[nodes["role"] == "output", "node_index"].to_numpy()
    in_to_out = adj[np.ix_(input_idx, output_idx)]
    proc_to_out = adj[np.ix_(proc_idx, output_idx)]
    print(f"Direct input->output edges (T4/T5 -> DN, bypassing HS/VS): {in_to_out.nnz}")
    print(f"Processing->output edges (HS/VS -> DN): {proc_to_out.nnz}")

    if counts.get("input", 0) == 0 or counts.get("output", 0) == 0:
        raise SystemExit("Sanity check failed: missing input or output neurons.")
    if proc_to_out.nnz == 0 and in_to_out.nnz == 0:
        raise SystemExit(
            "Sanity check failed: output layer has no incoming edges from input/processing "
            "-- the circuit is disconnected, RL signal could never reach it."
        )


def plot_subgraph(nodes: pd.DataFrame, agg: pd.DataFrame, out_path: Path, max_nodes: int = 300):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    # HS/VS + DN are small; T4/T5 are huge (thousands), so for a readable
    # figure we plot all processing/output neurons plus a random sample of
    # input neurons that actually connect to them.
    proc_out_ids = set(nodes.loc[nodes["role"] != "input", "root_id"])
    connected_inputs = set(agg.loc[agg["post_root_id"].isin(proc_out_ids), "pre_root_id"]) | set(
        agg.loc[agg["pre_root_id"].isin(proc_out_ids), "post_root_id"]
    )
    input_ids = set(nodes.loc[nodes["role"] == "input", "root_id"]) & connected_inputs
    if len(input_ids) > max_nodes:
        rng = np.random.default_rng(0)
        candidates = np.array(sorted(input_ids), dtype=np.int64)
        chosen = rng.choice(len(candidates), size=max_nodes, replace=False)
        input_ids = set(candidates[chosen].tolist())

    keep_ids = proc_out_ids | input_ids
    sub_nodes = nodes[nodes["root_id"].isin(keep_ids)]
    sub_edges = agg[agg["pre_root_id"].isin(keep_ids) & agg["post_root_id"].isin(keep_ids)]

    g = nx.DiGraph()
    role_color = {"input": "#4C72B0", "processing": "#DD8452", "output": "#55A868"}
    # itertuples (not iterrows) to avoid pandas upcasting the huge int64
    # root_id to float64 when a row is mixed with float columns like "sign".
    for row in sub_nodes.itertuples(index=False):
        g.add_node(row.root_id, role=row.role)
    for row in sub_edges.itertuples(index=False):
        g.add_edge(row.pre_root_id, row.post_root_id)

    colors = [role_color[g.nodes[n]["role"]] for n in g.nodes]
    pos = nx.spring_layout(g, seed=0, k=0.3)

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=25, alpha=0.85)
    nx.draw_networkx_edges(g, pos, alpha=0.15, arrows=False)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=r)
        for r, c in role_color.items()
    ]
    plt.legend(handles=legend_handles, loc="upper right")
    plt.title("Optomotor circuit subgraph (T4/T5 -> HS/VS -> DN)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved subgraph visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/Downloads").expanduser())
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FlyWire tables from {args.data_dir} ...")
    neurons, visual_types, connections, labels = load_tables(args.data_dir)

    print("Building node table (T4/T5 input, HS/VS processing, DN output) ...")
    nodes = build_node_table(visual_types, connections, labels)

    print("Building signed adjacency matrix ...")
    adj, agg = build_adjacency(nodes, neurons, connections)

    sanity_check(nodes, adj)

    nodes_path = args.out_dir / "circuit_nodes.csv"
    adj_path = args.out_dir / "circuit_adjacency.npz"
    fig_path = args.out_dir / "circuit_subgraph.png"

    nodes.to_csv(nodes_path, index=False)
    sp.save_npz(adj_path, adj)
    print(f"\nSaved node metadata to {nodes_path}")
    print(f"Saved adjacency matrix to {adj_path}")

    plot_subgraph(nodes, agg, fig_path)


if __name__ == "__main__":
    main()
