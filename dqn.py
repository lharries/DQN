import random

import torch
import torch.nn as nn

class Policy:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = Model(self.observation_space.shape[0], self.action_space.n)

    def train(self, state):
        pass

    def compute_actions(self, observation):
        return self.model(observation)

class Model(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Model, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(observation_space, action_space)
        self.ReLU = nn.ReLU()

    def transform(self, obs):
        return torch.tensor(obs, dtype=torch.float)

    def forward(self, obs):
        obs = self.transform(obs)
        obs = self.fc1(obs)
        obs = self.ReLU(obs)
        return torch.argmax(obs).item()

def learn():
    pass