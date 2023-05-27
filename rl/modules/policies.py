import torch
import torch.nn as nn

from modules.distributions import SquashedNormal
from modules.subnetworks import MLP

LOG_SIG_MAX = 2
LOG_SIG_MIN = -4


class DeterministicActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, max_action=1.):
        super().__init__()

        self.trunk = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim
        )
        self.max_action = max_action

    def forward(self, state):
        a = self.trunk(state)
        return self.max_action * torch.tanh(a)


class StochasticActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()

        self.trunk = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
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

    def sample_n(self, obs, n_samples):
        return self.forward(obs).sample((n_samples,))
