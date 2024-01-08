import collections
import pathlib
import random

import h5py
import numpy as np
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


# class OfflineDataset(IterableDataset):
#     def __init__(self, path, capacity=0, t_len=50, prioritize_ends=True, device="cpu"):
#         super().__init__()
#         self._path = pathlib.Path(path)
#         self._capacity = capacity
#         self._t_len = t_len
#         self._prioritize_ends = prioritize_ends
#         self._random = np.random.RandomState()

#         self.episodes = self.load_episodes()
#         print(f'keys: {self.episodes[0].keys()}')
#         self._num_episodes = len(self.episodes)
#         self._num_steps = sum([len(e['action'])-1 for e in self.episodes.values()])

#         print(f'num_episodes: {self._num_episodes}, num_steps: {self._num_steps}')

#         self.device = torch.device(device)

#     def load_episodes(self):
#         filenames = sorted(self._path.glob('*.npz'))
#         print('Shuffling order of offline trajectories!')
#         random.Random(0).shuffle(filenames)
#         if self._capacity:
#             num_steps = 0
#             num_episodes = 0
#             for filename in reversed(filenames):
#                 length = int(str(filename).split('-')[-1][:-4])
#                 num_steps += length
#                 num_episodes += 1
#                 if num_steps >= self._capacity:
#                     break
#             filenames = filenames[-num_episodes:]
#         episodes = {}
#         for i, filename in enumerate(filenames):
#             try:
#                 with filename.open('rb') as f:
#                     episode = np.load(f)
#                     episode = {k: episode[k] for k in episode.keys()}
#                     episode['image'] = (episode['image'].transpose(0, 3, 1, 2))
#                     # Conversion for older versions of npz files.
#                     if 'is_terminal' not in episode:
#                         episode['is_terminal'] = episode['discount'] == 0.
#             except Exception as e:
#                 print(f'Could not load episode {str(filename)}: {e}')
#                 continue
#             episodes[i] = episode
#         return episodes

#     def _sample_sequence(self):
#         episodes = list(self.episodes.values())
#         episode = self._random.choice(episodes)
#         ep_len = len(episode['action'])
#         upper = ep_len - self._t_len + 1
#         if self._prioritize_ends:
#             upper += self._t_len
#         index = min(self._random.randint(upper), ep_len - self._t_len)
#         sequence = {
#             k: convert(v[index: index + self._t_len])
#             for k, v in episode.items() if not k.startswith('log_')}
#         sequence['is_first'] = np.zeros(len(sequence['action']), bool)
#         sequence['is_first'][0] = True
#         for k, v in sequence.items():
#             sequence[k] = torch.as_tensor(v, device=self.device)
#         return tuple(sequence.values())
    
#     def __iter__(self):
#         while True:
#             yield self._sample_sequence()

#     def sample_transitions(self, batch_size):
#         episodes = self._random.choice(self.episodes, size=batch_size)
#         chunk = collections.defaultdict(list)
#         for episode in episodes:
#             idx = self._random.randint(len(episode['action']))
#             for key, value in episode.items():
#                 chunk[key].append(value[idx])
#         chunk = {k: torch.as_tensor(np.concatenate(v), device=self.device) for k, v in chunk.items()}
#         return chunk


class OfflineDataset(Dataset):
    def __init__(self, path, seq_len=50, device="cpu"):
        super().__init__()
        self._path = pathlib.Path(path)
        self._seq_len = seq_len

        self.episodes = self.load_episodes()
        print(f'keys: {self.episodes[0].keys()}')

        # create a list of episode indices
        indices = []
        for ep_idx, episode in self.episodes.items():
            ep_len = len(episode['action']) - 1
            for i in range(ep_len - self._seq_len + 1):
                indices.append((ep_idx, i, i+self._seq_len))
        self.indices = indices

        self._num_episodes = len(self.episodes)
        self._num_steps = sum([len(e['action'])-1 for e in self.episodes.values()])
        self._ep_rewards = [e['reward'].sum() for e in self.episodes.values()]

        print(f'num_episodes: {self._num_episodes}, num_steps: {self._num_steps}, max_ep_rewards: {np.max(self._ep_rewards)}, mean_ep_rewards: {np.mean(self._ep_rewards)}')

        self.device = torch.device(device)

    def load_episodes(self):
        filenames = sorted(self._path.glob('*.npz'))
        random.Random(0).shuffle(filenames)
        episodes = {}
        for i, filename in enumerate(filenames):
            try:
                with filename.open('rb') as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
                    if episode['image'].shape[-1] == 3:
                        episode['image'] = (episode['image'].transpose(0, 3, 1, 2))
                    # Conversion for older versions of npz files.
                    if 'is_terminal' not in episode:
                        episode['is_terminal'] = episode['discount'] == 0.
            except Exception as e:
                print(f'Could not load episode {str(filename)}: {e}')
                continue
            episodes[i] = episode
        return episodes

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ep_idx, start, end = self.indices[idx]
        episode = self.episodes[ep_idx]
        sequence = {
            k: convert(v[start: end])
            for k, v in episode.items() if not k.startswith('log_')
        }
        sequence['is_first'] = np.zeros(len(sequence['action']), bool)
        sequence['is_first'][0] = True
        for k, v in sequence.items():
            sequence[k] = torch.as_tensor(v, device=self.device)
        return tuple(sequence.values())
    
    def get_episode(self, ep_idx):
        episode = self.episodes[ep_idx]
        sequence = {
            k: convert(v)
            for k, v in episode.items() if not k.startswith('log_')
        }
        sequence['is_first'] = np.zeros(len(sequence['action']), bool)
        sequence['is_first'][0] = True
        for k, v in sequence.items():
            sequence[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
        return tuple(sequence.values())
    
    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size)
        keys = self.episodes[0].keys()
        results = {k: [] for k in keys}
        for idx in indices:
            ep_idx, start, end = self.indices[idx]
            episode = self.episodes[ep_idx]
            sequence = {
                k: convert(v[start: end])
                for k, v in episode.items() if not k.startswith('log_')
            }
            sequence['is_first'] = np.zeros(len(sequence['action']), bool)
            sequence['is_first'][0] = True
            for k, v in sequence.items():
                results[k].append(v)
        for k, v in results.items():
            results[k] = torch.as_tensor(np.stack(v), device=self.device)
        return results.values()


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_shape: Tuple,
        state_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.states = np.zeros((self._max_size,) + self.state_shape, dtype=state_dtype)
        self.next_states = np.zeros((self._max_size,) + self.state_shape, dtype=state_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.states[self._ptr] = np.array(state).copy()
        self.next_states[self._ptr] = np.array(next_state).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(states)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.states[indexes] = np.array(states).copy()
        self.next_states[indexes] = np.array(next_states).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.states.mean(0, keepdims=True)
        std = self.states.std(0, keepdims=True) + eps
        self.states = (self.states - mean) / std
        self.next_states = (self.next_states - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "states": torch.tensor(self.states[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_states": torch.tensor(self.next_states[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "states": self.states[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_states": self.next_states[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
    
    def save(self, file_name: str) -> None:
        h5f = h5py.File(file_name, 'w')
        h5f.create_dataset('states', data=self.states[:self._size])
        h5f.create_dataset('actions', data=self.actions[:self._size])
        h5f.create_dataset('next_states', data=self.next_states[:self._size])
        h5f.create_dataset('terminals', data=self.terminals[:self._size])
        h5f.create_dataset('rewards', data=self.rewards[:self._size])
        h5f.close()

    def load(self, file_name):
        h5f = h5py.File(file_name, 'r')
        self.states = np.array(h5f['states']).astype(self.state_dtype)
        self.actions = np.array(h5f['actions']).astype(self.action_dtype)
        self.next_states = np.array(h5f['next_states']).astype(self.state_dtype)
        self.terminals = np.array(h5f['terminals']).astype(np.float32)
        self.rewards = np.array(h5f['rewards']).astype(np.float32)
        self._size = len(self.states)
        self._max_size = len(self.states)
        self._ptr = 0
        h5f.close()


if __name__ == '__main__':
    dataset = OfflineDataset(path='vd4rl/main/cheetah_run/expert/64px', seq_len=50, device='cpu')

    # loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # for batch in loader:
    #     print(batch[-1].sum())

