import torch

from modules.subnetworks import Discriminator
from utils.general_utils import AttrDict
from .sac import SAC


class AMP(SAC):
    '''Refer to https://arxiv.org/abs/2104.02180. '''
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):  
        super().__init__(env_params, sampler, agent_cfg)

        # build discriminator
        self.update_disc_epoch = agent_cfg.update_disc_epoch
        self.disc = Discriminator(input_dim=self.dimo+self.dimg, hidden_dim=256, device=self.device)
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=agent_cfg.disc_lr)

    def update_disc(self, expert_obs, policy_obs):
        metrics = AttrDict()
        
        # expert
        expert_d = self.disc.trunk(expert_obs)
        policy_d = self.disc.trunk(policy_obs)

        expert_loss = (expert_d - 1).pow(2).mean()
        policy_loss = (policy_d + 1).pow(2).mean()

        gail_loss = expert_loss + policy_loss
        grad_pen = self.disc.compute_grad_pen(expert_obs, policy_obs)

        loss = (gail_loss + grad_pen)

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        metrics.update(AttrDict(
            disc_expert_loss=expert_loss.item(),
            disc_policy_loss=policy_loss.item(),
            disc_grad_pen=grad_pen.item()
        ))
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        for i in range(self.update_disc_epoch):
            # Sample from replay buffer 
            policy_obs, _, _, _, _  = self.get_samples(replay_buffer)
            expert_obs, _, _, _, _ = self.get_samples(demo_buffer)
        
            # Update discriminator
            metrics.update(self.update_disc(expert_obs, policy_obs))

        metrics.update(super().update(replay_buffer, demo_buffer))
        return metrics