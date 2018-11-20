from dqn import Policy
import gym

def main():

    env = gym.make('CartPole-v1')
    observation = env.reset()
    done = False

    policy = Policy(env.action_space)

    # import pdb; pdb.set_trace()
    while not done:
        # env.render()
        action = policy.compute_actions(observation)
        observation, reward, done, info = env.step(action)
        print(observation)

    env.close()

main()

# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action