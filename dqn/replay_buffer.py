import random
from collections import namedtuple

import numpy as np


class ReplayBuffer:
    def __init__(self, obs_shape, max_size):
        self.observations = np.zeros((max_size, *obs_shape), dtype=np.float)
        self.new_observations = np.zeros((max_size, *obs_shape), dtype=np.float)
        self.rewards = np.zeros((max_size,), dtype=np.float)
        self.actions = np.zeros((max_size,), dtype=np.int)

        self.max_size = max_size
        self.idx = 0
        self.size = 0

    def add(self, observations, new_observations, rewards, actions):
        self.observations[self.idx] = observations
        self.new_observations[self.idx] = new_observations
        self.rewards[self.idx] = rewards
        self.actions[self.idx] = actions

        self.idx = (self.idx + 1) % self.max_size
        self.size = max(self.idx, self.size)

    def sample(self, batch_size):
        assert batch_size <= self.size, f'Error: Number of entries in the replay buffer ({self.size}) is smaller than the batch size ({batch_size})'
        batch = random.sample(range(self.size), batch_size)

        return self.observations[batch], self.new_observations[batch], self.rewards[batch], self.actions[batch]
