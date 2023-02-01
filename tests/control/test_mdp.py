"""
Very basic testing of MDP
"""

import numpy as np
import pytest


from restless.control import MDP


@pytest.mark.parametrize("transition_matrix, expected_fail",
                         [
                             ([0.5 * np.ones((2, 2))]*2, False),
                             ([0.5 * np.ones((2, 2)), np.array([[0.3, 0.7], [0.9, 0.1]])], False),
                             ([0.6 * np.ones((2, 2)), 0.4 * np.ones((2, 2))], True),
                             ([0.5 * np.ones((2, 2)), -0.4 * np.ones((2, 2))], True)

                         ]
                         )
def test_mdp_transition_structure(transition_matrix, expected_fail):
    """
    Tests if assertion is raised when needed
    """
    reward_function = [np.array([0, 1]), np.array([1, 1])]
    if not expected_fail:
        _ = MDP(2, 2, transition_matrix, reward_function)
    else:
        with pytest.raises(Exception):
            _ = MDP(2, 2, transition_matrix, reward_function)


@pytest.mark.parametrize("reward_function, expected_fail",
                         [
                             ([np.array([0, 1]), np.array([1, 1])], False),
                             ([np.array([0, 1])], True),
                             ([np.array([0, 1]), np.array([0, 1]), np.array([0])], True)
                         ]
                         )
def test_mdp_reward_structure(reward_function, expected_fail):
    transition_matrix = [0.5 * np.ones((2, 2))]*2
    if not expected_fail:
        _ = MDP(2, 2, transition_matrix, reward_function)
    else:
        with pytest.raises(Exception):
            _ = MDP(2, 2, transition_matrix, reward_function)


def test_aperiodicity_transform():
    transition_kernel = 2*[np.fliplr(np.eye(2))]
    reward_function = [np.array([0, 1]), np.array([1, 1])]

    mdp = MDP(2, 2, transition_kernel, reward_function)
    aperiodic_mdp = mdp.to_aperiodic_mdp(tau=0.1)

    for action in range(2):
        assert np.all(np.diag(aperiodic_mdp.transition_kernel[action])>0)