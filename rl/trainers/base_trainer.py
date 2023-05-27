from abc import abstractmethod
from pathlib import Path


class BaseTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.work_dir = Path(cfg.cwd)
        self._setup()

    @abstractmethod
    def _setup(self):
        '''Setup trainer'''
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        '''Training agent'''
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        '''Evaluating agent.'''
        raise NotImplementedError
