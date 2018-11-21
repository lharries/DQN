import gym
import numpy as np

from replay_buffer import ReplayBuffer


class Runner:
    def __init__(self, env, policy, nsteps=10, mini_batch_size=5):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.replay_buffer = ReplayBuffer(env.observation_space.shape, 20)
        self.mini_batch_size = mini_batch_size

    def sample_experiences(self):


        for _ in range(self.nsteps):

            observation = self.env.reset()
            done = False
            while not done:
                # self.env.render()
                action = self.policy.compute_actions(observation)
                new_observation, reward, done, info = self.env.step(action)
                self.replay_buffer.add(observation, new_observation, reward, action)
                observation = new_observation

            self.env.close()

    def train(self):
        epochs = 5
        for _ in range(epochs):
            self.replay_buffer.sample(self.mini_batch_size)

    def run(self):

        # fill the replay buffer with experiences
        self.sample_experiences()
        self.train()


        # print(self.replay_buffer.storage)
