import random

import numpy as np
import torch
import torch.nn as nn


class EpsilonGreedyPolicy:
    """
    Epsilon greedy policy

    Epsilon 0 => always follows policy, 1 => allows follows random action
    """

    def __init__(self, observation_space, action_space, epsilon=1.0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_function = QFunction(observation_space, action_space)
        self.epsilon = epsilon

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

    def get_model(self):
        return self.q_function.get_model()

    def get_q_function(self):
        return self.q_function


class QFunction:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = Model(self.observation_space[0], self.action_space)

    def compute_Q_values(self, observation):
        return self.model(observation)

    def compute_max_Q_value(self, observation):
        return torch.max(self.compute_Q_values(observation), dim=1)[0]

    def compute_action(self, observation):
        assert len(observation.shape) == 1
        return torch.argmax(self.compute_Q_values(observation)).item()

    def get_model(self):
        return self.model


class Model(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=256):
        super(Model, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(observation_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        self.ReLU = nn.ReLU()

        self.one_hot_actions = torch.eye(action_space)

    def transform(self, obs):
        return torch.tensor(obs, dtype=torch.float)

    def make_one_hot(self, action):
        return self.one_hot_actions[action]

    def forward(self, obs):
        x = self.transform(obs)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        return x


def calc_loss(current_policy, target_policy, mini_batch, discount_factor=0.99):
    observations = torch.tensor(mini_batch["observations"], dtype=torch.float)
    new_observations = torch.tensor(mini_batch["new_observations"], dtype=torch.float)
    rewards = torch.tensor(mini_batch["rewards"], dtype=torch.float)
    actions = torch.tensor(mini_batch["actions"], dtype=torch.long)
    dones = torch.tensor(mini_batch["dones"], dtype=torch.float)

    # get the predicted q values of the states given the actions taken, using the current policy
    all_predicted_q_observations = current_policy.get_q_function().compute_Q_values(observations)
    indices_of_actions = torch.arange(len(all_predicted_q_observations), dtype=torch.long) * 2 + actions
    predicted_q_of_observation = torch.take(all_predicted_q_observations, indices_of_actions)

    # get the maximum q values of the next states given the best action
    predicted_q_of_next_obs = target_policy.get_q_function().compute_max_Q_value(new_observations)
    discount_pred_next_obs = discount_factor * predicted_q_of_next_obs

    # if done, use only the reward recieved, if not done, discount the future rewards and add to current rewards.
    # mask those which are done and add the rewards
    reward_and_discount_pred_next_obs = discount_pred_next_obs * (1 - dones) + rewards

    # calculate the loss
    loss = ((reward_and_discount_pred_next_obs - predicted_q_of_observation) ** 2)
    loss = loss.mean()
    return loss
