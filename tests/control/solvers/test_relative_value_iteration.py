"""
Testing discounted value iteration
"""

import numpy as np
import pytest

from restless.control import MDP, relative_value_iteration
from .test_discounted_value_iteration import random_transition_kernel


@pytest.mark.parametrize(
    "n_states, n_actions, precision",
    [(2, 2, 1e-3), (2, 3, 1e-4), (4, 4, 1e-3), (50, 10, 1e-3)],
)
def test_bellman_equation_valid(n_states, n_actions, precision):
    """
    Checks that the returned value vector checks the Bellman optimality equation
    """
    reward_function = [np.random.normal(size=(n_states,)) for _ in range(n_actions)]
    transition_kernel = random_transition_kernel(n_states, n_actions)
    mdp = MDP(n_states, n_actions, transition_kernel, reward_function)

    gain, bias = relative_value_iteration(mdp, precision, return_bias=True)

    bellman_equation = (
        gain * np.ones((n_states,))
        + bias
        - np.array(
            [
                np.max(
                    [
                        mdp.reward_function[action] + mdp.transition_kernel[action] @ bias
                        for action in range(mdp.n_actions)
                    ],
                    axis=0,
                )
            ]
        )
    ).reshape((mdp.n_states,))

    np.testing.assert_array_almost_equal(
        bellman_equation, np.zeros((n_states,)), decimal=np.floor(np.log10(0.5 / precision))
    )


@pytest.mark.parametrize(
    "discount, reward",
    [
        (0.1, 1),
        (0.5, 2),
        (0.9, 1),
        (0.95, 2),
    ],
)
def test_relative_value_iteration_constant_reward(discount, reward):
    """
    All state/actions pairs have the same reward --> we know the optimal value function
    """
    transition_matrix = random_transition_kernel(2, 2)
    reward_vector = 2 * [reward * np.ones((2,))]
    mdp = MDP(2, 2, transition_matrix, reward_vector)

    np.testing.assert_array_almost_equal(relative_value_iteration(mdp, 1e-3), reward * np.ones((2,)), decimal=3)
