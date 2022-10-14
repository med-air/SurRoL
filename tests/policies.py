import numpy as np
import torch
import torch.nn as nn
from rl.utils.distributions import SquashedNormal
from rl.utils.distributions import Bernoulli, Categorical, DiagGaussian
from rl.utils.utils import init

LOGITS_SCALE = 10
LOG_SIG_MAX = 2
LOG_SIG_MIN = -4

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), 
        )

    def forward(self, input):
        return self.mlp(input)


class DeterministicActor(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim=256):
        super().__init__()

        self.trunk = MLP(
            in_dim=dimo+dimg,
            out_dim=dima,
            hidden_dim=hidden_dim
        )

    def forward(self, obs):
        a = self.trunk(obs)
        return torch.tanh(a)


class StochasticActor(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim=256):
        super().__init__()

        self.trunk = MLP(
            in_dim=dimo+dimg,
            out_dim=2*dima,
            hidden_dim=hidden_dim
        )

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = LOG_SIG_MIN + 0.5 * (
            LOG_SIG_MAX - LOG_SIG_MIN
        ) * (log_std + 1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim):
        super().__init__()

        self.q = MLP(
            in_dim=dimo+dimg+dima,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q = self.q(sa)
        return q


class DoubleCritic(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim):
        super().__init__()
        
        self.q1 = MLP(
            in_dim=dimo+dimg+dima,
            out_dim=1,
            hidden_dim=hidden_dim
        )
        self.q2 = MLP(
            in_dim=dimo+dimg+dima,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

