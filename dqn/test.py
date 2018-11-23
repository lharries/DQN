import numpy as np

from replay_buffer import ReplayBuffer

from dqn import Model
from dqn import EpsilonGreedyPolicy

import gym
import torch
from dqn import calc_loss


def test_replay_buffer_adding():
    replay_buffer = ReplayBuffer((4,), 5)
    observations = np.ones((4,), dtype=np.float)
    new_observations = np.ones((4,), dtype=np.float) * 2
    dones = [False, False, False, False, True, False, False, False, False, True]

    assert observations.sum() == 4
    assert new_observations.sum() == 8

    for i in range(10):
        replay_buffer.add(observations * i, new_observations * i, i, i * 2, dones[i])

    # observations
    for i in range(5):
        value = i + 5
        assert replay_buffer.observations[i].sum() == value * 4

    # new_observations
    for i in range(5):
        value = i + 5
        assert replay_buffer.new_observations[i].sum() == value * 4 * 2

    # rewards
    for i in range(5):
        assert replay_buffer.rewards[i] == i + 5

    # actions
    for i in range(5):
        assert replay_buffer.actions[i] == (i + 5) * 2

    assert replay_buffer.dones.sum() == 1


def test_make_one_hot_actions():
    # 3 actions
    model = Model(4, 3)

    assert np.array_equal(model.make_one_hot(0), np.array([1, 0, 0]))
    assert np.array_equal(model.make_one_hot(1), np.array([0, 1, 0]))
    assert np.array_equal(model.make_one_hot(2), np.array([0, 0, 1]))

    # 5 actions
    model = Model(4, 5)

    assert np.array_equal(model.make_one_hot(0), np.array([1, 0, 0, 0, 0]))
    assert np.array_equal(model.make_one_hot(1), np.array([0, 1, 0, 0, 0]))
    assert np.array_equal(model.make_one_hot(2), np.array([0, 0, 1, 0, 0]))
    assert np.array_equal(model.make_one_hot(3), np.array([0, 0, 0, 1, 0]))
    assert np.array_equal(model.make_one_hot(4), np.array([0, 0, 0, 0, 1]))


def test_calc_loss():
    # def __init__(self, env_name='CartPole-v1', epochs=5, nsteps=5e6, mini_batch_size=32, replay_buffer_size=1000000,
    #              target_update_nsteps=1000000):
    env = gym.make("CartPole-v1")

    epsilon = 0.0

    policy = EpsilonGreedyPolicy(env.observation_space.shape, env.action_space.n, epsilon)
    target_policy = EpsilonGreedyPolicy(env.observation_space.shape, env.action_space.n,
                                        epsilon)

    optimizer = torch.optim.RMSprop(policy.get_model().parameters(), lr=0.00025, momentum=0.95)

    replay_buffer = create_mock_replay_buffer(env.observation_space.shape)

    mini_batch = replay_buffer.sample(32)

    for i in range(100):
        optimizer.zero_grad()

        loss = calc_loss(policy, mini_batch)
        print(loss)
        loss.backward()
        optimizer.step()
        # target_policy.get_model().load_state_dict(policy.get_model().state_dict())

    #
    # if step % target_update_nsteps == 0:
    #     update_policies()
    #
    # # anneal the epsilon value
    # if epsilon > epsilon_final:
    #     epsilon -= epsilon_anneal_rate
    #     policy.set_epsilon(epsilon)
    #     target_policy.set_epsilon(epsilon)


def create_mock_replay_buffer(shape, size=50):
    replay_buffer = ReplayBuffer(shape, size)
    # observations, new_observations, rewards, actions, done
    obs = np.array([0.2, 0.4, 0.1, 0.6])
    sample = (obs, obs, 1., 1, False)
    for _ in range(40):
        replay_buffer.add(*sample)

    return replay_buffer
