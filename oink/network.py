"""
Phase 2 — connectome-constrained network and its size-matched baseline.

The real circuit (from Phase 1) only has three edge types with meaningful
density: input->input (T4/T5 recurrence), input->processing (T4/T5 -> HS/VS),
and processing->output (HS/VS -> DN), plus a handful of others. We implement
exactly those as dense-but-masked linear layers -- dense is fine since the
largest one (12245 x 38) is under half a million entries.

A small trainable "retina" projects the environment's low-dim observation
into the input-neuron population, and a small trainable "motor" layer reads
the output-neuron population back down into action logits / a value. Those
two projections are the only parts of the network that have no biological
grounding -- they exist purely to interface a toy game with a brain circuit
sized for real vision, and are the same for both the connectome network and
the baseline so neither one is advantaged by them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn


class MaskedLinear(nn.Module):
    """A linear layer whose weights are constrained to a fixed sparsity mask.

    Dense: fine for the small submatrices (processing<->output, ~thousands of
    entries), where materializing the mask costs nothing.
    """

    def __init__(self, mask: torch.Tensor, init_weight: torch.Tensor | None = None):
        super().__init__()
        out_features, in_features = mask.shape
        weight = torch.zeros(out_features, in_features)
        if init_weight is not None:
            weight = init_weight.clone()
        else:
            nn.init.uniform_(weight, -0.1, 0.1)
            weight = weight * mask
        self.weight = nn.Parameter(weight)
        self.register_buffer("mask", mask)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ (self.weight * self.mask).T + self.bias


class SparseMaskedLinear(nn.Module):
    """Same idea as MaskedLinear but for layers too big to materialize dense.

    The real T4/T5 recurrent layer is 12245x12245 (150M entries) but only
    ~6800 of those are real synapses (0.005% density) -- a dense mask wastes
    ~30,000x the necessary compute per forward pass. This stores only the
    nonzero entries and does a genuine sparse matmul instead.
    """

    def __init__(self, shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray, values: np.ndarray | None = None):
        super().__init__()
        self.shape = shape
        self.register_buffer("indices", torch.tensor(np.stack([rows, cols]), dtype=torch.long))
        if values is not None:
            init = torch.tensor(values, dtype=torch.float32)
        else:
            init = torch.empty(len(rows)).uniform_(-0.1, 0.1)
        self.values = nn.Parameter(init)
        self.bias = nn.Parameter(torch.zeros(shape[0]))

    def forward(self, x):
        weight = torch.sparse_coo_tensor(self.indices, self.values, self.shape)
        return torch.sparse.mm(weight, x.T).T + self.bias


class CircuitTrunk(nn.Module):
    """input -> (recurrent input) -> processing -> output, each edge masked."""

    def __init__(
        self,
        n_input: int,
        n_processing: int,
        n_output: int,
        input_recurrent: "SparseMaskedLinear",
        input_proc_mask: torch.Tensor,
        proc_output_mask: torch.Tensor,
        input_proc_init=None,
        proc_output_init=None,
    ):
        super().__init__()
        self.n_input = n_input
        self.input_recurrent = input_recurrent
        self.input_to_proc = MaskedLinear(input_proc_mask, input_proc_init)
        self.proc_to_output = MaskedLinear(proc_output_mask, proc_output_init)
        self.act = nn.Tanh()

    def forward(self, input_activity):
        input_activity = self.act(input_activity + self.input_recurrent(input_activity))
        proc_activity = self.act(self.input_to_proc(input_activity))
        output_activity = self.act(self.proc_to_output(proc_activity))
        return output_activity


class ConnectomeNetwork(nn.Module):
    """Full trunk + trainable sensor/motor interface, wired from the real circuit."""

    def __init__(self, nodes_csv: Path, adjacency_npz: Path, obs_dim: int, out_dim: int):
        super().__init__()
        nodes = pd.read_csv(nodes_csv)
        adj = sp.load_npz(adjacency_npz).tocsr()

        idx = {role: nodes.loc[nodes["role"] == role, "node_index"].to_numpy() for role in ("input", "processing", "output")}
        self.n_input, self.n_processing, self.n_output = len(idx["input"]), len(idx["processing"]), len(idx["output"])

        def dense_submatrix(pre_idx, post_idx):
            dense = np.asarray(adj[np.ix_(pre_idx, post_idx)].todense(), dtype=np.float32)
            mask = torch.tensor((dense != 0).astype(np.float32).T)  # MaskedLinear is (out, in)
            weight = torch.tensor(dense.T) * 0.05  # synapse counts rescaled to a sane RL init
            return mask, weight

        def sparse_submatrix(pre_idx, post_idx):
            # (out, in) = (post, pre); never materializes a dense n_pre x n_post array.
            sub = adj[np.ix_(pre_idx, post_idx)].tocoo()
            weights = (sub.data.astype(np.float32)) * 0.05
            return SparseMaskedLinear((len(post_idx), len(pre_idx)), sub.col, sub.row, weights)

        input_recurrent = sparse_submatrix(idx["input"], idx["input"])
        ip_mask, ip_w = dense_submatrix(idx["input"], idx["processing"])
        po_mask, po_w = dense_submatrix(idx["processing"], idx["output"])

        self.trunk = CircuitTrunk(
            self.n_input, self.n_processing, self.n_output,
            input_recurrent, ip_mask, po_mask,
            ip_w, po_w,
        )
        self.sensor = nn.Linear(obs_dim, self.n_input)
        self.motor = nn.Linear(self.n_output, out_dim)

    def forward(self, obs):
        input_activity = torch.tanh(self.sensor(obs))
        output_activity = self.trunk(input_activity)
        return self.motor(output_activity)


class BaselineNetwork(nn.Module):
    """Same neuron counts and same total edge count per layer, randomly rewired."""

    def __init__(self, nodes_csv: Path, adjacency_npz: Path, obs_dim: int, out_dim: int, seed: int = 0):
        super().__init__()
        nodes = pd.read_csv(nodes_csv)
        adj = sp.load_npz(adjacency_npz).tocsr()
        idx = {role: nodes.loc[nodes["role"] == role, "node_index"].to_numpy() for role in ("input", "processing", "output")}
        n_input, n_processing, n_output = len(idx["input"]), len(idx["processing"]), len(idx["output"])
        self.n_input, self.n_processing, self.n_output = n_input, n_processing, n_output

        rng = np.random.default_rng(seed)

        def random_dense_mask(pre_idx, post_idx, n_pre, n_post):
            real_edges = adj[np.ix_(pre_idx, post_idx)].nnz
            mask = np.zeros((n_pre, n_post), dtype=np.float32)
            if real_edges > 0:
                flat = rng.choice(n_pre * n_post, size=real_edges, replace=False)
                mask.flat[flat] = 1.0
            return torch.tensor(mask.T)  # (out, in)

        def random_sparse(n_pre, n_post, n_edges):
            # Sample distinct (row, col) pairs without ever materializing the
            # n_pre x n_post space -- at ~0.005% density collisions are rare,
            # so a small over-sample + dedupe is cheap and exact enough.
            seen = set()
            while len(seen) < n_edges:
                need = n_edges - len(seen)
                rows = rng.integers(0, n_post, size=int(need * 1.05) + 1)
                cols = rng.integers(0, n_pre, size=int(need * 1.05) + 1)
                seen.update(zip(rows.tolist(), cols.tolist()))
            pairs = list(seen)[:n_edges]
            rows = np.array([p[0] for p in pairs])
            cols = np.array([p[1] for p in pairs])
            return SparseMaskedLinear((n_post, n_pre), rows, cols)

        n_ii_edges = adj[np.ix_(idx["input"], idx["input"])].nnz
        input_recurrent = random_sparse(n_input, n_input, n_ii_edges)
        ip_mask = random_dense_mask(idx["input"], idx["processing"], n_input, n_processing)
        po_mask = random_dense_mask(idx["processing"], idx["output"], n_processing, n_output)

        self.trunk = CircuitTrunk(n_input, n_processing, n_output, input_recurrent, ip_mask, po_mask)
        self.sensor = nn.Linear(obs_dim, n_input)
        self.motor = nn.Linear(n_output, out_dim)

    def forward(self, obs):
        input_activity = torch.tanh(self.sensor(obs))
        output_activity = self.trunk(input_activity)
        return self.motor(output_activity)
