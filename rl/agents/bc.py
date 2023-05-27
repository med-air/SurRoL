import copy

import numpy as np
import torch
import torch.nn.functional as F

from components.normalizer import Normalizer
from modules.critics import Critic
from modules.policies import DeterministicActor
from utils.general_utils import AttrDict
from .ddpgbc import DDPGBC


class BC(DDPGBC):
    def update_actor(self, obs, action):
        metrics = dict()

        action_out = self.actor(obs)
        bc_loss = -self.norm_dist(action_out, action).mean()

        # Optimize actor loss
        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()

        metrics['bc_loss'] = bc_loss.item()
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        for i in range(self.update_epoch):
            # sample from replay buffer 
            obs, action, reward, done, next_obs = self.get_samples(demo_buffer)
    
            # update critic and actor
            metrics.update(self.update_actor(obs, action))

        return metrics