import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, default_convert, default_collate
from typing import Optional, TypeVar
from collections import namedtuple
import numpy as np
import random
import pytorch_lightning as pl
from .parse_d4rl import parse_pickle_datasets, DATASET_DIR
from typing import Dict
from omegaconf import DictConfig

GOAL_DIMS = {
    'maze': [0, 1],
    'kitchen': [11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
}

T = TypeVar('T')
Batch = namedtuple('Batch', ['observations', 'actions', 'goal', 'valid_length', 'padding_mask'])


def obs_to_goal(env_name, obs):
    if 'maze' in env_name:  # antmaze provides goal as (x,y), the first two dimensions in the state space
        goal = obs[GOAL_DIMS['maze']]
    elif 'kitchen' in env_name:  # zero out redundant dimensions such as robot proprioceptive state
        goal_mask = np.ones(30, dtype=np.bool8)
        goal_mask[GOAL_DIMS['kitchen']] = False
        goal = np.where(goal_mask, 0., obs)  # already removed the last 30 dimension during preprocessing
    return goal


def preprocess_d4rl_episodes_from_path(path: str, max_episode_length: int, number: Optional[int] = None, proportion: Optional[float] = None):
    '''
    read dataset from pickle file
    '''
    with open(path, 'rb') as f:
        episodes = pickle.load(f)

    n_episode = len(episodes)
    
    print(f'Loading dataset from {path}: {n_episode} episodes')

    episode_lengths = [e['rewards'].shape[0] for e in episodes]
    key_dims = {key: episodes[0][key].shape[-1] for key in episodes[0].keys()}
    buffers = {key: np.zeros((n_episode, max_episode_length, key_dims[key]), dtype=np.float32) for key in episodes[0].keys()}

    for idx, e in enumerate(episodes):  # put episodes into fix sized numpy arrays (for padding)
        for key, buffer in buffers.items():
            buffer[idx, :episode_lengths[idx]] = e[key]
    
    buffers['episode_lengths'] = np.array(episode_lengths)

    return buffers


class SequenceDataset(Dataset):
    def __init__(self, env_name: str, epi_buffers: Dict, max_episode_length: int = 1000, ctx_size: int = 100, goal_type: str = 'state'):
        super().__init__()
        self.env_name = env_name
        self.max_episode_length = max_episode_length  # max length for each episode (env setting)
        self.ctx_size = ctx_size  # context window size
        self.goal_type = goal_type
        
        self.epi_buffers = epi_buffers
        self.traj_indices = self.sample_trajs_from_episode()

    def sample_trajs_from_episode(self):
        '''
            makes indices for sampling from dataset;
            each index maps to a trajectory (start, end)
        '''
        indices = []
        for i, epi_length in enumerate(self.epi_buffers['episode_lengths']):
            max_start = epi_length - self.ctx_size
            for start in range(max_start):
                end = start + self.ctx_size
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.traj_indices)

    def __getitem__(self, idx: int):
        epi_idx, start, end = self.traj_indices[idx]

        epi_length = self.epi_buffers['episode_lengths'][epi_idx]
        observations = self.epi_buffers['observations'][epi_idx, start:end]
        actions = self.epi_buffers['actions'][epi_idx, start:end]

        valid_length = min(epi_length, end) - start

        if self.goal_type == 'state':
            goal_idx = random.choice(range(end, epi_length))   # randomly sample a goal from the sequence after context window
            goal = self.epi_buffers['observations'][epi_idx, goal_idx]
            goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        else:
            goal = self.epi_buffers['avg_rtgs'][epi_idx, start:end]

        padding_mask = np.zeros(shape=(self.ctx_size, ), dtype=np.bool8)  # only consider one modality length
        padding_mask[valid_length:] = True

        batch = Batch(observations, actions, goal, valid_length, padding_mask)

        return batch


class PretrainDataset(SequenceDataset):
    def sample_trajs_from_episode(self):
        '''
            makes start indices for sampling from dataset
        '''
        indices = []
        for i, epi_length in enumerate(self.epi_buffers['episode_lengths']):
            max_start = epi_length - self.ctx_size
            for start in range(max_start):
                indices.append((i, start))
        indices = np.array(indices)
        return indices

    def __getitem__(self, idx: int):
        epi_idx, start = self.traj_indices[idx]
        epi_length = self.epi_buffers['episode_lengths'][epi_idx]

        end = random.choice(range(start + self.ctx_size, epi_length))
        valid_length = end - start

        observations = self.epi_buffers['observations'][epi_idx, start:end]
        actions = self.epi_buffers['actions'][epi_idx, start:end]

        if self.goal_type == 'state':
            goal = self.epi_buffers['observations'][epi_idx, end]
            goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        else:
            goal = self.epi_buffers['avg_rtgs'][epi_idx, start:start + self.ctx_size]

        return observations, actions, goal, valid_length


def pt_collate_fn(batch):
    batch_size = len(batch)
    obss = default_convert([item[0] for item in batch])
    acts = default_convert([item[1] for item in batch])
    goal = default_collate([item[2] for item in batch])
    valid_lengths = [item[3] for item in batch]

    max_valid_length = max(valid_lengths)
    pad_observations = torch.zeros(batch_size, max_valid_length, obss[0].shape[-1])
    pad_actions = torch.zeros(batch_size, max_valid_length, acts[0].shape[-1])
    padding_mask = torch.zeros(batch_size, max_valid_length, dtype=torch.bool)
    for idx, item in enumerate(zip(obss, acts, valid_lengths)):
        obs, act, valid_len = item
        pad_observations[idx, :valid_len] = obs
        pad_actions[idx, :valid_len] = act
        padding_mask[idx, valid_len:] = True
    valid_lengths = torch.tensor(valid_lengths)
    batch = Batch(pad_observations, pad_actions, goal, valid_lengths, padding_mask)
    return batch


class D4RLDataModule(pl.LightningDataModule):
    def __init__(self, goal_type, config: DictConfig):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.ctx_size = config.exp.ctx_size
        self.max_episode_length = config.max_episode_length
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.goal_type = goal_type
        self.dataset_path = os.path.join(DATASET_DIR, f'{self.env_name}.pkl')

    # def prepare_data(self):  # may comment out if not needed
    #     parse_pickle_datasets(self.env_name, self.dataset_path)

    def setup(self, stage: str):
        epi_buffer = preprocess_d4rl_episodes_from_path(self.dataset_path, self.max_episode_length)
        # split train/val and put into sequencedataset or goaldataset
        self._obs_dim = epi_buffer['observations'].shape[-1]
        self._action_dim = epi_buffer['actions'].shape[-1]

        num_epis = epi_buffer['rewards'].shape[0]
        if self.train_size <= 1:
            num_train = int(num_epis * self.train_size)
        else:
            num_train = self.train_size
        indices = np.arange(num_epis)

        train_indices = indices[:num_train]
        val_indices = indices[num_train: num_epis]
        train_buffer = {key: value[train_indices] for key, value in epi_buffer.items()}
        val_buffer = {key: value[val_indices] for key, value in epi_buffer.items()}

        if stage == 'trl':  # trajectory representation learning
            self.train = PretrainDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = PretrainDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
        else:
            self.train = SequenceDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = SequenceDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)


    def train_dataloader(self):
        if isinstance(self.train, PretrainDataset):
            return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.val, PretrainDataset):
            return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def get_obs_dim(self):
        return self._obs_dim
    
    def get_action_dim(self):
        return self._action_dim