import numpy as np
import os


actions = []
observations = []
infos = []

folder = "./needle_pick/"
file_names = os.listdir(folder)

for name in file_names:
    episode_info = np.load(folder+name,allow_pickle=True)
    actions.append(episode_info["acs"].squeeze())
    observations.append(episode_info["obs"].squeeze())
    infos.append(episode_info["info"].squeeze())


np.savez_compressed("./data_NeedlePick_random_150_demo1.npz", acs=actions, obs=observations, info=infos)  # save the file
