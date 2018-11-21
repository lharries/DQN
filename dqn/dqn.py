import random

import torch
import torch.nn as nn


class EpsilonGreedyPolicy:
    """
    Epsilon greedy policy

    Epsilon 0 => always follows policy, 1 => allows follows random action
    """

    def __init__(self, observation_space, action_space, epsilon=0.0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_function = QFunction(observation_space, action_space)
        self.epsilon = epsilon

    def train(self, state):
        raise ValueError("not yet implemented")

    def set_epsilon(self, epsilon):
        assert 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon

    def compute_actions(self, observation):
        if random.random() < self.epsilon:
            # random action
            return random.randint(0, self.action_space - 1)
        else:
            # use policy
            return self.q_function.compute_action(observation)


class QFunction:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = Model(self.observation_space[0], self.action_space)

    def compute_Q_values(self, observation):
        return self.model(observation)

    def compute_max_Q_value(self, observation):
        return max(self.compute_Q_values(observation)).item()

    def compute_action(self, observation):
        return torch.argmax(self.compute_Q_values(observation)).item()


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
