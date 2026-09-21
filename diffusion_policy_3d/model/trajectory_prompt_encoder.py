"""Encoder for the object-trajectory prompt.

Input is a fixed-length object trajectory (``prompt_length`` poses of
``state_dim`` = 7: xyz + quaternion), output is a single prompt vector that is
concatenated to the observation feature and used as diffusion conditioning.

Deliberately a plain MLP: whether a sequence architecture helps is not the
research question at this stage, so the prompt is simply flattened.
"""

import torch
import torch.nn as nn
from termcolor import cprint


class TrajectoryPromptEncoder(nn.Module):
    def __init__(self,
                 prompt_length=16,
                 state_dim=7,
                 prompt_dim=128,
                 hidden_dim=256,
                 n_hidden_layers=2):
        super().__init__()
        self.prompt_length = prompt_length
        self.state_dim = state_dim
        self.prompt_dim = prompt_dim

        in_dim = prompt_length * state_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, n_hidden_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, prompt_dim)]
        self.net = nn.Sequential(*layers)

        cprint(f"[TrajectoryPromptEncoder] {prompt_length}x{state_dim} -> {prompt_dim}", "yellow")

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """traj: (B, prompt_length, state_dim) -> (B, prompt_dim)"""
        assert traj.shape[-2:] == (self.prompt_length, self.state_dim), \
            f"expected (B, {self.prompt_length}, {self.state_dim}), got {tuple(traj.shape)}"
        B = traj.shape[0]
        return self.net(traj.reshape(B, -1))

    def output_shape(self):
        return self.prompt_dim
