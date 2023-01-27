import numpy as np
import pytest

from restless.envs.arms.channel import Channel


@pytest.mark.parametrize("p, q",
                         [
                             (0.8, 0.6),
                             (0.3, 0.8),
                             (0.9, 0.1)
                         ]
)
def test_transition_distribution(p, q):
    """
    Tests the one-step transition by Monte-Carlo
    """
    channel = Channel(p, q, init_state=1)
    n_trials = 10_000
    p_estimate = 0
    q_estimate = 0
    for _ in range(n_trials):
        channel.state = 1
        channel.step()
        p_estimate += 1 / n_trials if channel.state == 1 else 0
        channel.state = 0
        channel.step()
        q_estimate += 1 / n_trials if channel.state == 1 else 0

    np.testing.assert_almost_equal(p_estimate, p, decimal=2)
    np.testing.assert_almost_equal(q_estimate, q, decimal=2)


@pytest.mark.parametrize("p, q",
                         [
                             (0.8, 0.6),
                             (0.3, 0.8),
                             (0.9, 0.1)
                         ]
)
def test_stationary_distribution(p,q):
    """
    Checks that the stationary distribution estimates and its closed form are close, via Monte Carlo
    """
    channel = Channel(p, q, init_state=1)
    n_steps = 100_000
    state_1_estimate = 0
    state_0_estimate = 0
    for i in range(n_steps):
        channel.step()
        if i < 100:
            continue  ## forget initialisation
        state_1_estimate += 1 / n_steps if channel.state == 1 else 0
        state_0_estimate += 1 / n_steps if channel.state == 0 else 0

    stationary_db = channel.stationary_distribution()
    np.testing.assert_almost_equal(stationary_db[0], state_0_estimate, decimal=2)
    np.testing.assert_almost_equal(stationary_db[1], state_1_estimate, decimal=2)
