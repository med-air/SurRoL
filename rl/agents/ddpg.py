import copy

import numpy as np
import torch
import torch.nn.functional as F

from components.normalizer import Normalizer
from modules.critics import Critic
from modules.policies import DeterministicActor
from utils.general_utils import AttrDict
from .base import BaseAgent


class DDPG(BaseAgent):
    '''Refer to https://arxiv.org/abs/1509.02971. '''
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):  
        super().__init__()

        self.discount = agent_cfg.discount
        self.reward_scale = agent_cfg.reward_scale
        self.update_epoch = agent_cfg.update_epoch
        self.sampler = sampler    # same as which in buffer
        self.device = agent_cfg.device

        self.noise_eps = agent_cfg.noise_eps
        self.soft_target_tau = agent_cfg.soft_target_tau

        self.clip_obs = agent_cfg.clip_obs
        self.norm_clip = agent_cfg.norm_clip
        self.norm_eps = agent_cfg.norm_eps

        self.dima = env_params['act']
        self.dimo, self.dimg = env_params['obs'], env_params['goal']

        self.max_action = env_params['max_action']
        self.act_sampler = env_params['act_rand_sampler']

        # normarlizer
        self.o_norm = Normalizer(
            size=self.dimo, 
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )
        self.g_norm = Normalizer(
            size=self.dimg, 
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )

        # build policy
        self.actor = DeterministicActor(
            self.dimo+self.dimg, self.dima, agent_cfg.hidden_dim
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)

        self.critic = Critic(
            self.dimo+self.dimg+self.dima, agent_cfg.hidden_dim
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        # optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )

    def get_action(self, state, noise=False):
        with torch.no_grad():
            o, g = state['observation'], state['desired_goal']
            input_tensor = self._preproc_inputs(o, g)
            action = self.actor(input_tensor).cpu().data.numpy().flatten()

            # Gaussian noise
            if noise:
                action = (action + self.max_action * self.noise_eps * np.random.randn(action.shape[0])).clip(
                    -self.max_action, self.max_action)

        return action

    def update_critic(self, obs, action, reward, next_obs):
        with torch.no_grad():
            action_out = self.actor_target(next_obs)
            target_V = self.critic_target(next_obs, action_out)
            target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

            clip_return = 1 / (1 - self.discount)
            target_Q = torch.clamp(target_Q, -clip_return, 0).detach()

        Q = self.critic(obs, action)
        critic_loss = F.mse_loss(Q, target_Q)

        # optimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()      
        
        metrics = AttrDict(
            critic_q=Q.mean().item(),
            critic_target_q=target_Q.mean().item(),
            critic_loss=critic_loss.item(),
            bacth_reward=reward.mean().item()
        )
        return metrics

    def update_actor(self, obs):
        metrics = dict()

        action_out = self.actor(obs)
        Q_out = self.critic(obs, action_out)
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
    
            # update critic and actor
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor(obs))

            # update target critic and actor
            self.update_target()
        return metrics

    def update_target(self):
        # update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)
