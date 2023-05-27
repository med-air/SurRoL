import copy

import numpy as np
import torch
import torch.nn.functional as F

from components.normalizer import Normalizer
from modules.critics import DoubleCritic
from modules.policies import StochasticActor
from utils.general_utils import AttrDict
from .base import BaseAgent


class SAC(BaseAgent):
    '''Refer to https://arxiv.org/abs/1801.01290. '''
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

        self.learnable_temperature = agent_cfg.learnable_temperature
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
        self.actor = StochasticActor(
            self.dimo+self.dimg, self.dima*2, agent_cfg.hidden_dim
        ).to(agent_cfg.device)

        self.critic = DoubleCritic(
            self.dimo+self.dimg+self.dima, agent_cfg.hidden_dim
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        # entropy term 
        if self.learnable_temperature:
            self.target_entropy = -self.dima
            self.log_alpha = torch.tensor(np.log(agent_cfg.init_temperature)).to(self.device)
            self.log_alpha.requires_grad = True
        else:
            raise NotImplementedError

        # optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=agent_cfg.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state, noise=False):
        with torch.no_grad():
            o, g = state['observation'], state['desired_goal']
            input_tensor = self._preproc_inputs(o, g)
            dist = self.actor(input_tensor)
            if noise:
                action = dist.sample()
            else:
                action = dist.mean

        return action.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs):
        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_action_out = next_dist.rsample()

            log_prob = next_dist.log_prob(next_action_out).sum(-1, keepdim=True)
            target_V = self.critic_target.q(next_obs, next_action_out) - (self.alpha.detach() * log_prob)
            target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()      
        
        metrics = AttrDict(
            critic_q=Q1.mean().item(),
            critic_target_q=target_Q.mean().item(),
            critic_loss=critic_loss.item(),
            bacth_reward=reward.mean().item()
        )
        return metrics

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q = self.critic.q(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()
        actor_loss += action.pow(2).mean()

        # optimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # optimize alpha loss
        self.log_alpha_optimizer.zero_grad()
        if self.learnable_temperature:
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        else:
            raise NotImplementedError
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        metrics = AttrDict(
            actor_loss=actor_loss.item(),
            log_prob=log_prob.mean().item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item()
        )
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        for i in range(self.update_epoch):
            # sample from replay buffer 
            obs, action, reward, done, next_obs = self.get_samples(replay_buffer)
    
            # update critic and actor
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor_and_alpha(obs))

            # update target critic and actor
            self.update_target()
        return metrics

    def update_target(self):
        # update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)