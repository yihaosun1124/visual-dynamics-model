import collections
import pathlib
import random

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset
from collections import deque
from dm_env import StepType

from typing import Optional, Union, Tuple, Dict


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


class OfflineDataset(Dataset):
    def __init__(self, path, seq_len=20, device="cpu"):
        super().__init__()
        self._path = pathlib.Path(path)
        self._seq_len = seq_len

        # read language descriptions
        self.lang_descs= list(pd.read_csv(self._path / 'labels.csv', header=None).iloc[:,1][1:])

        # create a list of episode indices
        self.indices = list(range(len(self.lang_descs)))

        self._num_episodes = len(self.indices)

        print(f'num_episodes: {self._num_episodes}')

        self.device = torch.device(device)
        
    def load_episode(self, ep_idx):
        episode = np.load(self._path / f'trajs/{ep_idx}.npz')
        imgs = episode['imgs']
        if imgs.shape[-1] == 3:
            imgs = imgs.transpose(0, 3, 1, 2)
        imgs = torch.as_tensor(imgs, dtype=torch.uint8, device=self.device)
        actions = torch.as_tensor(episode['actions'], dtype=torch.float32, device=self.device).float()
        rewards = torch.zeros(len(actions), dtype=torch.float32, device=self.device)
        terminals = torch.zeros(len(actions), device=self.device).bool()
        return imgs, actions, rewards, terminals

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        imgs, actions, rewards, terminals = self.load_episode(idx)
        lang_desc = self.lang_descs[idx]
        return imgs, actions, rewards, terminals, lang_desc
    
    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size)
        results = {k: [] for k in ["imgs", "actions", "rewards", "terminals"]}
        for idx in indices:
            imgs, actions, rewards, terminals = self.load_episode(idx)
            results["imgs"].append(imgs)
            results["actions"].append(actions)
            results["rewards"].append(rewards)
            results["terminals"].append(terminals)
        for k, v in results.items():
            results[k] = torch.stack(v)
        langs = [self.lang_descs[idx] for idx in indices]
        return results["imgs"], results["actions"], results["rewards"], results["terminals"], langs


if __name__ == '__main__':
    # dataset = OfflineDataset(path='/home/yihaosun/code/rl/visual-dynamics-model/generated_data', seq_len=20, device='cpu')
    # print(dataset.lang_descs[-1])

    # loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # for batch in loader:
    #     print(batch[0].shape)

    hf5 = h5py.File('/home/yihaosun/code/rl/visual-dynamics-model/generated_data/data.hdf5', 'r')
    save_path = '/home/yihaosun/code/rl/visual-dynamics-model/generated_data/trajs/'
    imgs = hf5['sim']['imgs'][:]
    actions = hf5['sim']['actions'][:]
    ep_num = imgs.shape[0]
    print(ep_num)
    from tqdm import trange
    for i in trange(ep_num):
        episode = {'imgs': imgs[i].astype(np.uint8), 'actions': actions[i].astype(np.float32)}
        np.savez(save_path + f"{i}.npz", **episode)
    hf5.close()
