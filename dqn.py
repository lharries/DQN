import random

import torch
import torch.nn as nn

class Policy:
    def __init__(self, action_space):
        self.action_space = action_space

    def train(self, state):
        pass

    def compute_actions(self, observation):
        return self.action_space.sample()

class Model(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Model, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(observation_space, action_space)

    def forward(self, obs):
        return nn.ReLU(self.fc1(obs))

def learn():
    pass