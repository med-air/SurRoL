import torch
import torch.nn as nn
from torch import autograd


class MLP(nn.Module):
    '''We use 4-layer MLP as default'''
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


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()
        self.returns = None
        # self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         policy_state,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        alpha = alpha.expand_as(expert_state).to(expert_state.device)

        mixup_data = alpha * expert_state + (1 - alpha) * policy_state
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def predict_reward(self, state):
        with torch.no_grad():
            d = self.trunk(state)
            return torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)