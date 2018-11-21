import gym
import numpy as np

from replay_buffer import ReplayBuffer


class Runner:
    def __init__(self, env, policy, epochs=5, nsteps=200, mini_batch_size=32, replay_buffer_size=100):
        self.env = env
        self.policy = policy
        self.epochs = epochs
        self.nsteps = nsteps
        self.replay_buffer = ReplayBuffer(env.observation_space.shape, replay_buffer_size)
        self.mini_batch_size = mini_batch_size

    def run(self):
        print("training...\n")

        step = 0

        while True:

            # run episode
            observation = self.env.reset()
            done = False

            episode_reward = 0

            while not done:
                action = self.policy.compute_actions(observation)
                new_observation, reward, done, info = self.env.step(action)

                self.replay_buffer.add(observation, new_observation, reward, action)

                # train
                if step > 50:
                    mini_batch = self.replay_buffer.sample(self.mini_batch_size)
                # TODO: Update models

                episode_reward += reward
                step += 1

            print(f'Step number {step}, reward: {episode_reward}')

            if step >= self.nsteps:
                return