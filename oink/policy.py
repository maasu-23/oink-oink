"""Wires ConnectomeNetwork/BaselineNetwork into stable-baselines3 as a features extractor.

The trunk (connectome or baseline) produces the DN/output-layer activity;
SB3's default actor/critic linear heads sit on top of that, identical for
both conditions so the only difference between runs is the trunk wiring.
"""
from pathlib import Path

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from oink.network import BaselineNetwork, ConnectomeNetwork


class CircuitFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, nodes_csv: Path, adjacency_npz: Path, variant: str = "connectome", seed: int = 0):
        if variant == "connectome":
            net = ConnectomeNetwork(nodes_csv, adjacency_npz, obs_dim=observation_space.shape[0], out_dim=observation_space.shape[0])
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
