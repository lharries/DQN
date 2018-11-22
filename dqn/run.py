import gym
import numpy as np

from replay_buffer import ReplayBuffer

from dqn import EpsilonGreedyPolicy


class Runner:
    def __init__(self, env_name='CartPole-v1', epochs=5, nsteps=2e3, mini_batch_size=32, replay_buffer_size=100):
        self.env = gym.make(env_name)
        self.policy = EpsilonGreedyPolicy(self.env.observation_space.shape, self.env.action_space.n)
        self.epochs = epochs
        self.nsteps = nsteps
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, replay_buffer_size)
        self.mini_batch_size = mini_batch_size

        # probability you choose a random action instead of following the policy
        self.epsilon = 1.0
        self.epsilon_final = 0.1
        self.epsilon_anneal_steps = 1e3
        self.epsilon_anneal_rate = (self.epsilon - self.epsilon_final) / self.epsilon_anneal_steps

    def run(self):
        print("training...\n")

        step = 0

        # each loop represents sampling one episode and training
        while True:

            observation = self.env.reset()
            episode_done = False
            episode_reward = 0

            while not episode_done:
                action = self.policy.compute_actions(observation)
                new_observation, reward, episode_done, info = self.env.step(action)

                self.replay_buffer.add(observation, new_observation, reward, action, episode_done)

                # train
                if step > 50:
                    mini_batch = self.replay_buffer.sample(self.mini_batch_size)
                    # TODO: Update models

                # anneal the self.epsilon value
                if self.epsilon > self.epsilon_final:
                    self.epsilon -= self.epsilon_anneal_rate

                episode_reward += reward
                step += 1

            print(f'Step number {step}, reward: {episode_reward}')

            if step >= self.nsteps:
                return


def main():
    runner = Runner()
    runner.run()


main()
