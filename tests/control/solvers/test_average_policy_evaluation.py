"""
Testing averaged policy evaluation routine on toy MDPs
TODO validty of Bellman, compare to closed-form, comparison with discount
"""

import numpy as np
import pytest

from restless.control.solvers.policy_evaluation import (
    average_reward_policy_evaluation,
    get_reward_vector,
    get_transition_matrix,
)
from restless.control import MDP, Policy


@pytest.mark.parametrize(
    "policy_actions, expect_error",
    [([0, 0], False), ([0, 1], False), ([1, 0], True), ([1, 1], False)],
)
def test_error_raised_when_non_constant_gain(policy_actions, expect_error):
    """
    Checks that policy that induce more than 1 irreducible class raise error
    Here,  with a simple two-state MDP
    """
    transition_kernel = [np.array([[0.2, 0.8], [0, 1]]), np.array([[1, 0], [0.3, 0.7]])]
    reward_function = 2 * [
        np.ones(
            2,
        )
    ]  # not important here
    mdp = MDP(2, 2, transition_kernel, reward_function)
    pi = Policy(policy_actions)

    if not expect_error:
        print(
            average_reward_policy_evaluation(pi, mdp, check_nb_irreducible_class=True)
        )
    else:
        with pytest.raises(Exception):
            _ = average_reward_policy_evaluation(
                pi, mdp, check_nb_irreducible_class=True
            )


@pytest.mark.parametrize(
    "policy_actions, reward, expected_gain",
    [
        ([0, 0], 1, 1),
        ([0, 1], 1, 1),
        ([1, 0], 0, 0),
    ],
)
def test_gain_value(policy_actions, reward, expected_gain):
    """
    Checks that the gain of different policy, on toy two-state MDPs where each action has the same reward
    """
    transition_kernel = 2 * [
        0.5 * np.ones((2, 2))
    ]  ## actions have no impact on transitions
    reward_function = 2 * [reward * np.ones(2)]
    mdp = MDP(2, 2, transition_kernel, reward_function)
    pi = Policy(policy_actions)

    gain = average_reward_policy_evaluation(pi, mdp, return_bias=False)
    np.testing.assert_almost_equal(gain, expected_gain)


def random_ergodic_chain(n_states, n_actions):
    """
    Returns a random MDP transition kernel that is ergodic under any policy
    """
    eps = 1e-2
    transition_kernel = []
    for _ in range(n_actions):
        transition_matrix = np.random.uniform(eps, 1, (n_states, n_states))
        normalized_transition_matrix = (
            transition_matrix / transition_matrix.sum(axis=1)[:, np.newaxis]
        )
        transition_kernel.append(normalized_transition_matrix)
    return transition_kernel


@pytest.mark.parametrize("n_states, n_actions", [(2, 2), (2, 3), (4, 4), (5, 2)])
def test_bellman_equation_valid(n_states, n_actions):
    """
    Checks that the returned (gain, bias) couple checks the Bellman policy evaluation equation
    """
    reward_function = [np.random.normal(size=(n_states)) for _ in range(n_actions)]
    transition_kernel = random_ergodic_chain(n_states, n_actions)
    pi = Policy([np.random.randint(0, n_actions) for _ in range(n_states)])
    mdp = MDP(n_states, n_actions, transition_kernel, reward_function)

    gain, bias = average_reward_policy_evaluation(pi, mdp)
    pi_reward_vector = get_reward_vector(pi, mdp)
    pi_transition_matrix = get_transition_matrix(pi, mdp)

    bellman_equation = (
        gain * np.ones((n_states,))
        + bias
        - pi_transition_matrix @ bias
        - pi_reward_vector
    )

    np.testing.assert_array_almost_equal(bellman_equation, np.zeros((n_states,)))
