import random
from collections import namedtuple

Experience = namedtuple("Experience", ["observations", "rewards", "actions"])


class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = list()

    def add(self, observations, rewards, actions):
        experience = Experience(observations, rewards, actions)

        self.storage.append(experience)

    def sample(self):
        random.choice(self.storage)
