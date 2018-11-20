import gym
import numpy as np

from replay_buffer import ReplayBuffer


class Runner:
    def __init__(self, env, policy, nsteps=5):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.replay_buffer = ReplayBuffer(10)

    def sample_experiences(self):


        for _ in range(self.nsteps):

            observation = self.env.reset()
            done = False
            while not done:
                # self.env.render()
                action = self.policy.compute_actions(observation)
                observation, reward, done, info = self.env.step(action)
                self.replay_buffer.add(observation, reward, action)

            self.env.close()

    # def train(self):
    #     epochs = 5
    #     for _ in range(epochs)

    def run(self):

        # fill the replay buffer with experiences
        self.sample_experiences()



        print(self.replay_buffer.storage)
