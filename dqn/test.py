import numpy as np

from replay_buffer import ReplayBuffer

from dqn import Model


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
