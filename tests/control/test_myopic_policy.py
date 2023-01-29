"""
Testing myopic policy's construction
"""

import numpy as np
import pytest

from restless.control import MyopicPolicy
from restless.control import MDP


@pytest.mark.parametrize(
    "reward_vector_0, reward_vector_1, expected_result",
    [
        ([2, 0], [2, 0], [0, 0]),
        ([2, 1], [1, 3], [0, 1]),
        ([2, 4], [3, 3], [1, 0])
    ]
)
def test_myopic_policy(reward_vector_0, reward_vector_1, expected_result):
    """
    Checks that myopic policy's constructor is indeed myopic
    (i.e. checking that the policy is indeed myopic on a toy MDP)
    """
    n_states = 2
    n_actions = 2

    transition_matrix = [0.5*np.ones((n_states, n_states)) for _ in range(n_actions)]
    reward_function = [np.array(reward_vector_0), np.array(reward_vector_1)]

    mdp = MDP(n_states, n_actions, transition_matrix, reward_function)
    myopic = MyopicPolicy(mdp)

    assert myopic.actions == expected_result

