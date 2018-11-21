import numpy as np

from replay_buffer import ReplayBuffer

class Test:
    def test_replay_buffer(self):
        replay_buffer = ReplayBuffer((4,), 5)
        observations = np.ones((4,), dtype=np.float)
        new_observations = np.ones((4,), dtype=np.float) * 2

        assert observations.sum() == 4
        assert new_observations.sum() == 8

        for i in range(10):
            replay_buffer.add(observations*i, new_observations*i, i, i*2)

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

def test_test():
    assert 1 == 1