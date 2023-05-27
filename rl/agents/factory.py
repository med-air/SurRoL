from .ddpg import DDPG
from .ddpgbc import DDPGBC
from .col import CoL
from .dex import DEX

from .sac import SAC
from .sqil import SQIL
from .amp import AMP
from .awac import AWAC

AGENTS = {
    'DDPG': DDPG,
    'DDPGBC': DDPGBC,
    'CoL': CoL,
    'DEX': DEX,
    'SAC': SAC,
    'SQIL': SQIL,
    'AMP': AMP,
    'AWAC': AWAC
}


def make_agent(env_params, sampler, cfg):
    if cfg.name not in AGENTS.keys():
        assert 'Agent is not supported: %s' % cfg.name
    else:
        return AGENTS[cfg.name](
            env_params=env_params,
            sampler=sampler,
            agent_cfg=cfg
        )
