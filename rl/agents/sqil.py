import torch

from .sac import SAC


class SQIL(SAC):
    '''Refer to https://arxiv.org/abs/1905.11108. '''
    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        for i in range(self.update_epoch):
            # sample from replay buffer 
            obs, action, reward, done, next_obs = self.get_samples(replay_buffer)
            reward = torch.zeros_like(reward)   # 0 reward for self-collected data

            # update critic and actor
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor_and_alpha(obs))

            # sample from demo buffer 
            obs, action, reward, done, next_obs = self.get_samples(demo_buffer)
            reward = torch.ones_like(reward)    # 1 reward for demonstration deta

            # update critic and actor
            self.update_critic(obs, action, reward, next_obs)
            self.update_actor_and_alpha(obs)

            # update target critic and actor
            self.update_target()
        return metrics