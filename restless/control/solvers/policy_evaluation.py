"""
Policy evaluation for finite horizon, discounted and average MDPs
TODO test vectorial representation
"""
from typing import Tuple

import numpy as np

from restless.control import MDP, DiscountedMDP, FiniteHorizonMDP, Policy


def get_reward_vector(pi: Policy, mdp: MDP) -> np.array:
    # # Sets up the vector based representation of the policy's reward vector on some MDP
    reward_vector = np.array([mdp.reward_function[pi.actions[state]][state] for state in range(mdp.n_states)])
    return reward_vector


def get_transition_matrix(pi: Policy, mdp: MDP) -> np.array:
    # Instantaneous Transition Matrix
    transition_matrix = np.vstack(
        [mdp.transition_kernel[pi.actions[state]][state, :] for state in range(mdp.n_states)]
    )
    return transition_matrix


def policy_evaluation(pi: Policy, mdp: MDP) -> np.array:
    """
    Runs policy evaluation given the MDP type. Returns the value function as a vector.
    """
    if isinstance(mdp, DiscountedMDP):
        return discounted_policy_evaluation(pi, mdp)
    elif isinstance(mdp, FiniteHorizonMDP):
        raise NotImplementedError  # not done yet

    # by default the MDP is assumed to be evaluated under average-reward
    return average_reward_policy_evaluation(pi, mdp)


def discounted_policy_evaluation(pi: Policy, mdp: DiscountedMDP) -> np.array:
    lambd = mdp.discount
    transition_matrix = get_transition_matrix(pi, mdp)
    reward_vector = get_reward_vector(pi, mdp)
    return np.linalg.solve(np.eye(mdp.n_states) - lambd * transition_matrix, reward_vector)


def average_reward_policy_evaluation(pi: Policy, mdp: MDP) -> np.array:
    # TODO
    raise NotImplementedError
