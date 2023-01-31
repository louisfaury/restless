"""
Testing discounted policy evaluation routine on toy MDPs
"""

import numpy as np
import pytest

from restless.control.solvers.policy_evaluation import discounted_policy_evaluation
from restless.control import DiscountedMDP, Policy


@pytest.mark.parametrize(
    "discount, reward, policy_actions",
    [
        (0.1, 1, [0, 0]),
        (0.5, 2, [1, 0]),
        (0.9, 1, [1, 1]),
        (0.95, 2, [0, 1]),
    ],
)
def test_policy_evaluation_constant_reward(discount, reward, policy_actions):
    """
    All state/actions pairs have the same reward --> every policy should have the same discounted total reward
    """
    transition_matrix = 2 * [0.5 * np.ones((2, 2))]
    reward_vector = 2 * [reward * np.ones((2,))]
    mdp = DiscountedMDP(2, 2, transition_matrix, reward_vector, discount)
    pi = Policy(policy_actions)

    np.testing.assert_array_almost_equal(discounted_policy_evaluation(pi, mdp), (reward / (1 - discount)) * np.ones((2,)))


@pytest.mark.parametrize(
    "discount, reward_vector_0, reward_vector_1, policy_actions, expected",
    [
        (0.1, [0, 1], [0, 2], [0, 1], [0, 2/(1-0.1)]),
        (0.6, [1, 1], [3, 2], [0, 0], [1/(1-0.6), 1/(1-0.6)]),
        (0.95, [0, 3], [1, 2], [1, 0], [1/(1-0.95), 3/(1-0.95)])
    ]
)
def test_policy_evaluation_non_communicating(discount, reward_vector_0, reward_vector_1, policy_actions, expected):
    """
    Toy experiments with independent states
    """
    transition_matrix = 2*[np.eye(2)]
    reward_function = [np.array(reward_vector_0), np.array(reward_vector_1)]
    mdp = DiscountedMDP(2, 2, transition_matrix, reward_function, discount)

    pi = Policy(policy_actions)

    np.testing.assert_array_almost_equal(discounted_policy_evaluation(pi, mdp), expected)
