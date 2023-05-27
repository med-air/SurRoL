import os

import numpy as np

from .general_utils import AttrDict, RecursiveAverageMeter


def get_env_params(env, cfg):
    obs = env.reset()
    env_params = AttrDict(
        obs=obs['observation'].shape[0],
        achieved_goal=obs['achieved_goal'].shape[0],
        goal=obs['desired_goal'].shape[0],
        act=env.action_space.shape[0],
        act_rand_sampler=env.action_space.sample,
        max_timesteps=env._max_episode_steps,
        max_action=env.action_space.high[0],
    )
    return env_params


class ReplayCache:
    def __init__(self, T):
        self.T = T
        self.reset()

    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones = [], [], [], [], []

    def store_transition(self, obs, action, done):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)

    def store_obs(self, obs):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])

    def pop(self):
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        #print(self.ag)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)

        self.reset()
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones)
        return episode

    
def init_buffer(cfg, buffer, agent, normalize=True):
    '''Load demonstrations into buffer and initilaize normalizer'''
    demo_path = cfg.demo_path
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs = demo['obs'], demo['acs']

    episode_cache = ReplayCache(buffer.T)
    for epsd in range(cfg.num_demo):
        episode_cache.store_obs(demo_obs[epsd][0])
        for i in range(buffer.T):
            print(buffer.T)
            episode_cache.store_transition(
                obs=demo_obs[epsd][i+1],
                action=demo_acs[epsd][i],
                done=i==(buffer.T-1),
            )
        episode = episode_cache.pop()
        buffer.store_episode(episode)
        if normalize:
            agent.update_normalizer(episode)


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""
    def __init__(self):
        self.rollouts = []

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts    # rollout storage should not be empty
        stats = RecursiveAverageMeter()
        for rollout in self.rollouts:
            stats.update(AttrDict(
                avg_reward=np.stack(rollout.reward).sum(),
                avg_success_rate=rollout.success[-1]
            ))
        return stats.avg

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]