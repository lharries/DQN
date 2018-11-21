from dqn import Policy
import gym
import argparse

from runner import Runner


def main(episodes=5, env_name='CartPole-v1'):
    env = gym.make(env_name)
    policy = Policy(env.observation_space, env.action_space)

    # import pdb; pdb.set_trace()
    runner = Runner(env, policy)

    runner.run()

    print("---done---")


main()

# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
