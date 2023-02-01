"""
Testing discounted value iteration
"""

import numpy as np
import pytest

from restless.control import DiscountedMDP, discounted_value_iterations


def random_transition_kernel(n_states, n_actions):
    """
    Returns a random MDP transition kernel
    """
    transition_kernel = []
    for _ in range(n_actions):
        transition_matrix = np.random.uniform(0, 1, (n_states, n_states))
        normalized_transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, np.newaxis]
        transition_kernel.append(normalized_transition_matrix)
    return transition_kernel


@pytest.mark.parametrize(
    "n_states, n_actions, discount, precision",
    [(2, 2, 0.9, 0.1), (2, 2, 0.9, 1e-3), (2, 3, 0.5, 1e-2), (4, 4, 0.95, 1e-2), (50, 10, 0.995, 1e-3)],
)
def test_bellman_equation_valid(n_states, n_actions, discount, precision):
    """
    Checks that the returned value vector checks the Bellman optimality equation
    """
    reward_function = [np.random.normal(size=(n_states,)) for _ in range(n_actions)]
    transition_kernel = random_transition_kernel(n_states, n_actions)
    mdp = DiscountedMDP(n_states, n_actions, transition_kernel, reward_function, discount)

    discounted_value = discounted_value_iterations(mdp, precision)

    bellman_equation = discounted_value - np.array(
        [
            np.max(
                [
                    mdp.reward_function[action] + mdp.discount * mdp.transition_kernel[action] @ discounted_value
                    for action in range(mdp.n_actions)
                ],
                axis=0,
            )
        ]
    ).reshape((mdp.n_states,))

    np.testing.assert_array_almost_equal(
        bellman_equation, np.zeros((n_states,)), decimal=np.floor(np.log10(1 / precision))
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
def test_value_iteration_constant_reward(discount, reward):
    """
    All state/actions pairs have the same reward --> we know the optimal value function
    """
    transition_matrix = 2 * [0.5 * np.ones((2, 2))]
    reward_vector = 2 * [reward * np.ones((2,))]
    mdp = DiscountedMDP(2, 2, transition_matrix, reward_vector, discount)

    np.testing.assert_array_almost_equal(
        discounted_value_iterations(mdp, 1e-3), (reward / (1 - discount)) * np.ones((2,)), decimal=3
    )
