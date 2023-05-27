from .ddpgbc import DDPGBC


class CoL(DDPGBC):
    '''Refer to https://arxiv.org/abs/1910.04281. '''
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):  
        super().__init__(env_params, sampler, agent_cfg)
        self.offline_updated = False
        self.offline_steps = agent_cfg.offline_steps

    def update_offline(self, demo_buffer):
        for i in range(self.offline_steps):
            obs, action, _, _, _ = self.get_samples(demo_buffer)

            action_out = self.actor(obs)
            bc_loss = - self.norm_dist(action_out, action).mean()

            self.actor_optimizer.zero_grad()
            bc_loss.backward()
            self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(param.data)

    def update_actor(self, obs, action, is_demo=False):
        metrics = dict()

        action_out = self.actor(obs)
        Q_out = self.critic(obs, action_out)

        # Refer to https://arxiv.org/pdf/1709.10089.pdf
        if is_demo:
            bc_loss = self.norm_dist(action_out, action)
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
        if not self.offline_updated:
            # offline pretraining
            self.update_offline(demo_buffer)
            self.offline_updated = True

        metrics = super().update(replay_buffer, demo_buffer)
        return metrics