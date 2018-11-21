import random

import torch
import torch.nn as nn


class Policy:
    def __init__(self, observation_space, action_space, epsilon=1.0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = Model(self.observation_space.shape[0], self.action_space.n)
        self.epsilon = epsilon

    def train(self, state):
        raise ValueError("not yet implemented")

    def set_epsilon(self, epsilon):
        assert 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon

    def compute_actions(self, observation):
        if random.random() < self.epsilon:
            # random action
            return random.randint(0, self.action_space)
        else:
            # use policy
            return torch.argmax(self.model(observation)).item()



class Model(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Model, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(observation_space, action_space)
        self.ReLU = nn.ReLU()

        self.one_hot_actions = torch.eye(action_space)

    def transform(self, obs):
        return torch.tensor(obs, dtype=torch.float)

    def make_one_hot(self, action):
        # TODO: convert this function to a util
        return self.one_hot_actions[action]

    def forward(self, obs):
        obs = self.transform(obs)
        x = self.transform(obs)
        x = self.fc1(x)
        x = self.ReLU(x)
        return x


def learn():
    pass
