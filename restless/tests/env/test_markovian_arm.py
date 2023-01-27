
import numpy as np
import pytest

from restless.envs.arms.markov_arm import MarkovArm
from restless.envs import Channel


@pytest.mark.parametrize("p, q",
                         [
                             (0.8, 0.6),
                             (0.3, 0.8),
                             (0.9, 0.1)
                         ]
)
def test_stationary_distribution_markov(p,q):
    """
    Checks the diagonalisation-based computation of the Markov arm
    Here, just compare to the closed-form we have in the channel case
    """
    channel = Channel(p, q)
    markov_arm = MarkovArm(2, channel.transition_matrix, channel.reward_vector)

    np.testing.assert_almost_equal(markov_arm.stationary_distribution(), channel.stationary_distribution())