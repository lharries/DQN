import gym
import numpy as np
import torch

from replay_buffer import ReplayBuffer

from dqn import EpsilonGreedyPolicy, calc_loss


class Runner:
    def __init__(self, env_name='CartPole-v0', nsteps=5e6, mini_batch_size=128, replay_buffer_size=10000,
                 target_update_nsteps=10):
        self.env = gym.make(env_name)

        self.epsilon = 0.9
        self.epsilon_final = 0.05
        self.epsilon_anneal_steps = 1e4
        self.epsilon_anneal_rate = (self.epsilon - self.epsilon_final) / self.epsilon_anneal_steps

        self.policy = EpsilonGreedyPolicy(self.env.observation_space.shape, self.env.action_space.n, self.epsilon)
        self.target_policy = EpsilonGreedyPolicy(self.env.observation_space.shape, self.env.action_space.n,
                                                 self.epsilon)
        self.update_policies()

        self.nsteps = nsteps
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, replay_buffer_size)
        self.mini_batch_size = mini_batch_size
        self.target_update_nsteps = target_update_nsteps

        # probability you choose a random action instead of following the policy
        self.sample_before_anneal = 3000

        self.optimizer = torch.optim.Adam(self.policy.get_model().parameters(), lr=1e-4)

    def update_policies(self):
        self.target_policy.get_model().load_state_dict(self.policy.get_model().state_dict())

    def run(self):
        print("training...\n")

        step = 0
        loss = None
        nepisode = 0

        # each loop represents sampling one episode and training
        while True:

            observation = self.env.reset()
            episode_done = False
            episode_reward = 0

            while not episode_done:
                action = self.policy.compute_actions(observation)
                new_observation, reward, episode_done, info = self.env.step(action)

                self.replay_buffer.add(observation, new_observation, reward, action, episode_done)

                observation = new_observation

                # train
                if step > self.sample_before_anneal:
                    mini_batch = self.replay_buffer.sample(self.mini_batch_size)

                    self.optimizer.zero_grad()

                    loss = calc_loss(self.policy, self.target_policy, mini_batch)

                    loss.backward()
                    self.optimizer.step()

                    # anneal the self.epsilon value
                    if self.epsilon > self.epsilon_final:
                        self.epsilon -= self.epsilon_anneal_rate
                        self.policy.set_epsilon(self.epsilon)
                        self.target_policy.set_epsilon(self.epsilon)

                episode_reward += reward
                step += 1

            nepisode += 1

            if nepisode % 50 == 0:
                print(f'Step number {step}, reward: {episode_reward}, loss: {loss}, epsilon: {self.epsilon}')

            if nepisode % 10 == 0:
                self.update_policies()
                print("updating the policy")

            if step >= self.nsteps:
                return


def main():
    runner = Runner()
    runner.run()


main()
