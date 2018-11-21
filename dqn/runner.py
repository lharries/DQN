import gym
import numpy as np

from replay_buffer import ReplayBuffer


class Runner:
    def __init__(self, env, policy, epochs=5, nsteps=10, mini_batch_size=5):
        self.env = env
        self.policy = policy
        self.epochs = epochs
        self.nsteps = nsteps
        self.replay_buffer = ReplayBuffer(env.observation_space.shape, 20)
        self.mini_batch_size = mini_batch_size

    def sample_experiences(self):

        episode_rewards = []

        for _ in range(self.nsteps):

            observation = self.env.reset()
            done = False

            episode_reward = 0

            while not done:
                action = self.policy.compute_actions(observation)
                new_observation, reward, done, info = self.env.step(action)

                self.replay_buffer.add(observation, new_observation, reward, action)

                observation = new_observation
                episode_reward += reward

            episode_rewards.append(episode_reward)

            self.env.close()

        average_episode_reward = sum(episode_rewards)/len(episode_rewards)
        return average_episode_reward

    def train(self):
        epochs = 5
        for _ in range(epochs):
            self.replay_buffer.sample(self.mini_batch_size)

    def run(self):
        print("training...\n")

        for epoch in range(self.epochs):
            average_episode_reward = self.sample_experiences()
            self.train()
            print("Epoch " + str(epoch) + ", average reward: " + str(average_episode_reward))
