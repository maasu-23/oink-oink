"""Wires ConnectomeNetwork/BaselineNetwork into stable-baselines3 as a features extractor.

The trunk (connectome or baseline) produces the DN/output-layer activity;
SB3's default actor/critic linear heads sit on top of that, identical for
both conditions so the only difference between runs is the trunk wiring.
"""
from pathlib import Path

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from oink.network import BaselineNetwork, ConnectomeNetwork

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_ppo(path, nodes_csv=None, adjacency_npz=None, **kwargs):
    """PPO.load that repoints the circuit files, so models trained elsewhere (SageMaker) load locally.

    SB3 stores the features-extractor kwargs -- including absolute paths to the
    circuit CSV/NPZ -- inside the zip and rebuilds the policy from them on load.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _, _ = load_from_zip_file(path, load_data=True, custom_objects=None, device="cpu")
    policy_kwargs = dict(data["policy_kwargs"])
    fe_kwargs = dict(policy_kwargs["features_extractor_kwargs"])
    fe_kwargs["nodes_csv"] = Path(nodes_csv or REPO_ROOT / "data/processed/circuit_nodes.csv")
    fe_kwargs["adjacency_npz"] = Path(adjacency_npz or REPO_ROOT / "data/processed/circuit_adjacency.npz")
    policy_kwargs["features_extractor_kwargs"] = fe_kwargs
    return PPO.load(path, custom_objects={"policy_kwargs": policy_kwargs}, **kwargs)


class CircuitFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, nodes_csv: Path, adjacency_npz: Path, variant: str = "connectome", seed: int = 0):
        if variant == "connectome":
            net = ConnectomeNetwork(nodes_csv, adjacency_npz, obs_dim=observation_space.shape[0], out_dim=observation_space.shape[0])
        elif variant == "connectome_randinit":  # ablation: real wiring, random initial weights
            net = ConnectomeNetwork(nodes_csv, adjacency_npz, obs_dim=observation_space.shape[0], out_dim=observation_space.shape[0], synapse_init=False)
        elif variant == "baseline":
            net = BaselineNetwork(nodes_csv, adjacency_npz, obs_dim=observation_space.shape[0], out_dim=observation_space.shape[0], seed=seed)
        else:
            raise ValueError(f"unknown variant {variant!r}")

        super().__init__(observation_space, features_dim=net.n_output)
        self.sensor = net.sensor
        self.trunk = net.trunk
        # motor head is unused here -- SB3 supplies its own actor/critic heads
        # on top of the raw output-neuron activity.
        del net.motor

    def forward(self, obs):
        import torch

        input_activity = torch.tanh(self.sensor(obs))
        return self.trunk(input_activity)
