import gym
import numpy as np
import torch
import argparse

from replay_buffer import ReplayBuffer
from dqn import EpsilonGreedyPolicy, Loss

from tensorboardX import SummaryWriter


class Runner:
    def __init__(self, env_name,
                 lr,
                 mini_batch_size,
                 replay_buffer_size,
                 target_update_nsteps,
                 max_episodes,
                 discount_factor,
                 epsilon_start,
                 epsilon_final,
                 epsilon_anneal_steps,
                 render=False):

        self.mini_batch_size = mini_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_nsteps = target_update_nsteps
        self.max_episodes = max_episodes
        self.discount_factor = discount_factor
        self.render = render

        self.env = gym.make(env_name)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, self.replay_buffer_size)

        # probability of choosing random actions
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.epsilon_anneal_rate = (self.epsilon - self.epsilon_final) / self.epsilon_anneal_steps

        self.sample_before_anneal = 128

        self.policy = EpsilonGreedyPolicy(self.env.observation_space.shape, self.env.action_space.n, self.epsilon)
        self.target_policy = EpsilonGreedyPolicy(self.env.observation_space.shape, self.env.action_space.n,
                                                 self.epsilon)
        self.loss = Loss(self.policy, self.target_policy, self.discount_factor)
        self.update_policies()

        self.optimizer = torch.optim.Adam(self.policy.get_model().parameters(), lr=lr)

    def update_policies(self):
        self.target_policy.get_model().load_state_dict(self.policy.get_model().state_dict())

    def run(self):
        print("training...\n")
        writer = SummaryWriter()

        step = 0
        loss = None
        nepisode = 0
        episode_rewards = []

        # each loop represents sampling one episode and training
        while True:
            # setup environment
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

                    loss = self.loss(mini_batch)

                    loss.backward()
                    self.optimizer.step()

                    # anneal the self.epsilon value
                    if self.epsilon > self.epsilon_final:
                        self.epsilon -= self.epsilon_anneal_rate
                        self.policy.set_epsilon(self.epsilon)
                        self.target_policy.set_epsilon(self.epsilon)

                episode_reward += reward
                step += 1

                if self.render:
                    self.env.render()

                writer.add_scalar('data/episode_number', nepisode, step)
                writer.add_scalar('data/step', step, step)
                writer.add_scalar('data/episode_reward', episode_reward, step)
                writer.add_scalar('data/epsilon', self.epsilon, step)
                if loss:
                    writer.add_scalar('data/loss', loss, step)

                writer.add_scalar('data/mini_batch_size', self.mini_batch_size, step)
                writer.add_scalar('data/replay_buffer_size', self.replay_buffer_size, step)
                writer.add_scalar('data/target_update_nsteps', self.target_update_nsteps, step)
                writer.add_scalar('data/max_episodes', self.max_episodes, step)
                writer.add_scalar('data/sample_before_anneal', self.sample_before_anneal, step)
                writer.add_scalar('data/buffer_size', self.replay_buffer.size, step)

            nepisode += 1
            episode_rewards.append(episode_reward)

            writer.add_scalar('data/episode_reward_average', sum(episode_rewards) / len(episode_rewards), step)
            
            if nepisode % 10 == 0:
                print(
                    f'Episode number \t {nepisode} \t'
                    f'Step number \t {step}, '
                    f'mean_reward: \t {sum(episode_rewards) / len(episode_rewards)}\t'
                    f'max_reward: \t {max(episode_rewards)}\t'
                    f'loss: \t {loss}\t'
                    f'epsilon: \t {self.epsilon}\t')

                episode_rewards = []

            if nepisode % 10 == 0:
                self.update_policies()

            if nepisode >= self.max_episodes:
                if self.render:
                    self.env.close()
                return


def main():
    parser = argparse.ArgumentParser(description="DQN")
    parser.add_argument("--env_name", default="CartPole-v1", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--mini_batch_size", default=128, type=int)
    parser.add_argument("--replay_buffer_size", default=1000, type=int)
    parser.add_argument("--target_update_nsteps", default=10, type=int)
    parser.add_argument("--max_episodes", default=200, type=int)
    parser.add_argument("--discount_factor", default=0.99, type=float)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--epsilon_start", default=0.9, type=int)
    parser.add_argument("--epsilon_final", default=0.05, type=int)
    parser.add_argument("--epsilon_anneal_steps", default=2000, type=int)
    args = parser.parse_args()
    print(vars(args))
    runner = Runner(**vars(args))
    runner.run()


main()
