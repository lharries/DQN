from dqn import EpsilonGreedyPolicy
import gym
import argparse

from runner import Runner


def main(episodes=5, env_name='CartPole-v1'):
    env = gym.make(env_name)
    policy = EpsilonGreedyPolicy(env.observation_space.shape, env.action_space.n)

    # import pdb; pdb.set_trace()
    runner = Runner(env, policy)

    runner.run()

    print("\n---done---")


main()

# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
