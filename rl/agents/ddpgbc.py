import numpy as np
import torch

from .ddpg import DDPG


class DDPGBC(DDPG):
    '''Refer to https://arxiv.org/abs/1709.10089. '''
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):  
        super().__init__(env_params, sampler, agent_cfg)

        self.aux_weight = agent_cfg.aux_weight
        self.p_dist = agent_cfg.p_dist

    def update_actor(self, obs, action, is_demo=False):
        metrics = dict()

        action_out = self.actor(obs)
        Q_out = self.critic(obs, action_out)

        # Refer to https://arxiv.org/pdf/1709.10089.pdf
        if is_demo:
            bc_loss = self.norm_dist(action_out, action)
            # Q-filter
            with torch.no_grad():
                q_filter = self.critic_target(obs, action) >= self.critic_target(obs, action_out)
            bc_loss = q_filter * bc_loss
            actor_loss = -(Q_out + self.aux_weight * bc_loss).mean()
        else:
            actor_loss = -(Q_out).mean()

        actor_loss += action_out.pow(2).mean()

        # Optimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        for i in range(self.update_epoch):
            # sample from replay buffer 
            obs, action, reward, done, next_obs = self.get_samples(replay_buffer)
    
            # ppdate critic and actor
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor(obs, action))

            # sample from demo buffer 
            obs, action, reward, done, next_obs = self.get_samples(demo_buffer)

            # ppdate critic and actor
            self.update_critic(obs, action, reward, next_obs)
            self.update_actor(obs, action, is_demo=True)

            # Update target critic and actor
            self.update_target()
        return metrics

    def norm_dist(self, a1, a2):
        #return - (a1 - a2).pow(2).sum(dim=1,keepdim=True) / self.action_dim
        self.p_dist = np.inf if self.p_dist == -1 else self.p_dist
        return - torch.norm(a1 - a2, p=self.p_dist, dim=1, keepdim=True).pow(2) / self.dima 