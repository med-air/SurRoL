from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn

from utils.mpi import sync_networks


class BaseAgent(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_action(self, state):
        '''Interact with the world'''
        raise NotImplementedError

    def update(self, state, noise=False):
        '''Update the agent'''
        raise NotImplementedError

    def get_samples(self, replay_buffer):
		# sample from replay buffer 
        transitions = replay_buffer.sample()

        # preprocess
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        obs = self.to_torch(inputs_norm)
        next_obs = self.to_torch(inputs_next_norm)
        action = self.to_torch(transitions['actions'])
        reward = self.to_torch(transitions['r'])
        done = self.to_torch(transitions['dones'])

        return obs, action, reward, done, next_obs

    def update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, dones = episode_batch.obs, episode_batch.ag, episode_batch.g, \
                                                    episode_batch.actions, episode_batch.dones
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.sampler.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        '''Preprocess obs'''
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preproc_inputs(self, o, g, dim=0, device=None):
        '''Normalize obs'''
        o_norm = self.o_norm.normalize(o, device=device)
        g_norm = self.g_norm.normalize(g, device=device)
 
        if isinstance(o_norm, np.ndarray):
            inputs = np.concatenate([o_norm, g_norm], dim)
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif isinstance(o_norm, torch.Tensor): 
            inputs = torch.cat([o_norm, g_norm], dim=1)
        return inputs

    def to_torch(self, array, copy=True):
        if copy:
            return torch.tensor(array, dtype=torch.float32).to(self.device)
        return torch.as_tensor(array).to(self.device)

    def sync_networks(self):
        sync_networks(self)