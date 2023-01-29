"""
Testing the vectorial representation built in the policy evaluation module
"""

import numpy as np
import pytest

from restless.control import MDP, Policy
from restless.control.solvers.policy_evaluation import get_reward_vector, get_transition_matrix


@pytest.mark.parametrize("reward_vector_0, reward_vector_1, policy_action, expected_result",
                         [
                             ([1, 0], [0, 1], [0, 0], [1, 0]),
                             ([1, 0], [2, 1], [1, 0], [2, 0]),
                             ([1, 0], [2, 1], [0, 1], [1, 1]),
                             ([-1, 2], [2, 2], [1, 0], [2, 2]),
                         ]
                         )
def test_reward_vector(reward_vector_0, reward_vector_1, policy_action, expected_result):
    transition_matrix = [0.5 * np.ones((2, 2))]*2
    reward_function = [np.array(reward_vector_0), np.array(reward_vector_1)]

    mdp = MDP(2, 2, transition_matrix, reward_function)

    pi = Policy(policy_action)
    np.testing.assert_array_equal(get_reward_vector(pi, mdp), np.array(expected_result))


@pytest.mark.parametrize("transition_matrix, policy_action, expected_result",
                         [
                             ([0.5*np.ones((2, 2))]*2, [0, 0], [[0.5, 0.5], [0.5, 0.5]]),
                             ([0.5*np.ones((2, 2))]*2, [0, 1], [[0.5, 0.5], [0.5, 0.5]]),
                             ([np.array([[1, 0], [0, 1]])]+[0.5*np.ones((2, 2))], [0, 0], [[1, 0], [0, 1]]),
                             ([np.array([[1, 0], [0, 1]])]+[0.5*np.ones((2, 2))], [0, 1], [[1, 0], [0.5, 0.5]]),
                             ([np.array([[1, 0], [0, 1]])]+[0.5*np.ones((2, 2))], [1, 0], [[0.5, 0.5], [0, 1]])
                         ]
                         )
def test_reward_vector(transition_matrix, policy_action, expected_result):
    reward_function = [np.array([1, 0]), np.array([1, 0])]

    mdp = MDP(2, 2, transition_matrix, reward_function)

    pi = Policy(policy_action)
    np.testing.assert_array_equal(get_transition_matrix(pi, mdp), np.array(expected_result))