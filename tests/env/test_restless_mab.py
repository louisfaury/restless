"""
Basic tests for restless MAB functionalities, through the ChannelAccess
"""

import numpy as np
import pytest

from restless.envs import ChannelAccessMAB


@pytest.mark.parametrize("n_arms, init_states",
                         [
                             (2, [0, 0]),
                             (3, [0, 1, 0])
                         ]
                         )
def test_reset(n_arms, init_states):
    mab = ChannelAccessMAB(n_arms=n_arms, p=0.8, q=0.4)
    mab.reset_state(init_states)
    np.testing.assert_array_equal(np.array([arm.state for arm in mab.arm_list]), np.array(init_states))
