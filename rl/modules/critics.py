import torch
import torch.nn as nn

from modules.subnetworks import MLP


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.q = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q = self.q(sa)
        return q


class DoubleCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.q1 = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

        self.q2 = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q(self, state, action):
        # for double q learning
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)