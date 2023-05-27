import glob
import os
import pipes
import sys

import numpy as np
import torch

from components.logger import logger
from utils.general_utils import get_last_argmax, str2int


class CheckpointHandler:
    @staticmethod
    def get_ckpt_name(episode):
        return 'weights_ep{}.pth'.format(episode)

    @staticmethod
    def get_episode(path):
        checkpoint_names = glob.glob(os.path.abspath(path) + "/*.pth")
        if len(checkpoint_names) == 0:
            logger.error("No checkpoints found at {}!".format(path))
        processed_names = [file.split('/')[-1].replace('weights_ep', '').replace('.pth', '')
                           for file in checkpoint_names]
        episodes = list(filter(lambda x: x is not None, [str2int(name) for name in processed_names]))
        return episodes
    
    @staticmethod
    def get_resume_ckpt_file(resume, path):
        if resume == 'latest':
            max_episode = np.max(CheckpointHandler.get_episode(path))
            resume_file = CheckpointHandler.get_ckpt_name(max_episode)
        elif resume == 'best':
            episodes = CheckpointHandler.get_episode(path)
            file_paths = [os.path.join(path, CheckpointHandler.get_ckpt_name(episode)) for episode in episodes]
            scores = [torch.load(file_path)['score'] for file_path in file_paths]
            max_episode = episodes[get_last_argmax(scores)]
            resume_file = CheckpointHandler.get_ckpt_name(max_episode)
            logger.info(f'Checkpoints with success rate {scores}, the highest success rate {max(scores)}!')
        return os.path.join(path, resume_file), max_episode

    @staticmethod
    def save_checkpoint(state, folder, filename='checkpoint.pth'):
        torch.save(state, os.path.join(folder, filename))

    @staticmethod
    def load_checkpoint(checkpt_dir, agent, device, episode='best'):
        """Loads weigths from checkpoint."""
        checkpt_path, max_episode = CheckpointHandler.get_resume_ckpt_file(episode, checkpt_dir)
        checkpt = torch.load(checkpt_path, map_location=device)
    
        logger.info(f'Loading pre-trained model from {checkpt_path}!')
        agent.load_state_dict(checkpt['state_dict'])
        agent.g_norm = checkpt['g_norm']
        agent.o_norm = checkpt['o_norm']


def save_cmd(base_dir):
  train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
  train_cmd += '\n\n'
  print('\n' + '*' * 80)
  print('Training command:\n' + train_cmd)
  print('*' * 80 + '\n')
  with open(os.path.join(base_dir, "cmd.txt"), "a") as f:
    f.write(train_cmd)
