from math import isnan

import torch
import torch.nn.functional as F

from utils.general_utils import AttrDict
from .sac import SAC


class AWAC(SAC):
    '''Refer to https://arxiv.org/abs/2006.09359. '''
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):  
        super().__init__(env_params, sampler, agent_cfg)

        # AWAC parameters
        self.n_action_samples = agent_cfg.n_action_samples
        self.lam = agent_cfg.lam
        self.offline_updated = False
        self.offline_steps = agent_cfg.offline_steps

    def update_offline(self, demo_buffer):
        for i in range(self.offline_steps):
            obs, action, reward, done, next_obs = self.get_samples(demo_buffer)

            # update critic and actor
            self.update_critic(obs, action, reward, next_obs)
            self.update_actor(obs, action)

            self.update_target()

    def update_critic(self, obs, action, reward, next_obs):
        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_action_out = next_dist.rsample()

            target_V = self.critic_target.q(next_obs, next_action_out)
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

    def update_actor(self, obs, action):
        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True).clamp(-30, 30)

        # compute exponential weight
        weights = self._compute_weights(obs, action)
        actor_loss = -(log_prob * weights).sum()
        actor_loss += action.pow(2).mean()

        # pathology of density based methods, refer to https://arxiv.org/abs/2212.13936.
        if not isnan(actor_loss.item()):
            # optimize actor loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        metrics = AttrDict(
            actor_loss=actor_loss.item(),
            log_prob=log_prob.mean().item(),
        )
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        if not self.offline_updated:
            # offline pretraining
            self.update_offline(demo_buffer)
            self.offline_updated = True

        for i in range(self.update_epoch):
            # sample from replay buffer 
            obs, action, reward, done, next_obs = self.get_samples(replay_buffer)
    
            # update critic and actor
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor(obs, action))

            # update target critic and actor
            self.update_target()
        return metrics

    def _compute_weights(self, obs, act):
        with torch.no_grad():
            batch_size = obs.shape[0]

            # compute action-value
            q_values = self.critic.q(obs, act)
            
            # sample actions
            policy_actions = self.actor.sample_n(obs, self.n_action_samples)
            flat_actions = policy_actions.reshape(-1, self.dima)

            # repeat observation
            reshaped_obs = obs.view(batch_size, 1, *obs.shape[1:])
            reshaped_obs = reshaped_obs.expand(batch_size, self.n_action_samples, *obs.shape[1:])
            flat_obs = reshaped_obs.reshape(-1, *obs.shape[1:])

            # compute state-value
            flat_v_values = self.critic.q(flat_obs, flat_actions)
            reshaped_v_values = flat_v_values.view(obs.shape[0], -1, 1)
            v_values = reshaped_v_values.mean(dim=1)

            # compute normalized weight
            adv_values = (q_values - v_values).view(-1)
            weights = F.softmax(adv_values / self.lam, dim=0).view(-1, 1)

        return weights * adv_values.numel()