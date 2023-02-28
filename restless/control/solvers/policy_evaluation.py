"""
Policy evaluation for finite horizon, discounted and average MDPs
"""
from typing import Tuple, Union

import numpy as np
from scipy.linalg import eigvals

from restless.control.mdp import MarkovDecisionProcess as MDP
from restless.control.mdp import DiscountedMarkovDecisionProcess as DiscountedMDP
from restless.control.policy import Policy


def get_reward_vector(pi: Policy, mdp: MDP) -> np.array:
    """
    Sets up the vector based representation of the policy's reward vector on some MDP
    """
    reward_vector = np.array([mdp.reward_function[pi.actions[state]][state] for state in range(mdp.n_states)])
    return reward_vector


def get_transition_matrix(pi: Policy, mdp: MDP) -> np.array:
    """
    Instantaneous Transition Matrix of a policy on some MDP
    """
    transition_matrix = np.vstack(
        [mdp.transition_kernel[pi.actions[state]][state, :] for state in range(mdp.n_states)]
    )
    return transition_matrix


def discounted_policy_evaluation(pi: Policy, mdp: DiscountedMDP) -> np.array:
    """

    Parameters
    ----------
    pi : Policy
        The policy to evaluate
    mdp : MarkovDecisionProcess

    Returns
    -------
    : np.array
        The policy `pi`'s value

    TODO
    ----
    Implement an iterative version (fixed-point solver) for large MDPs to avoid inverting enormous matrix.
    """
    lambd = mdp.discount
    transition_matrix = get_transition_matrix(pi, mdp)
    reward_vector = get_reward_vector(pi, mdp)
    return np.linalg.solve(np.eye(mdp.n_states) - lambd * transition_matrix, reward_vector)


def average_reward_policy_evaluation(
    pi: Policy, mdp: MDP, return_bias: bool = True, check_nb_irreducible_class: bool = False
) -> Union[float, Tuple[float, np.array]]:
    """
    Policy evaluation for an average-gain MDP

    Warnings
    --------
    Assumes that the policy's gain is constant,
    i.e that the policy induces a Markov chain with a single recurrent class.
    If unsure about this property, you can pass a parameter (`check_nb_irreducible_class`) that will trigger
    an error if the chain's has several irreducible class.

    Parameters
    ----------
    pi : Policy
        The policy to evaluate
    mdp : MarkovDecisionProcess
    return_bias : bool
        Whether to return the policy's differential value function
    check_nb_irreducible_class : bool
        If True, checks that the policy's induces a Markov Chain with a unique irreductible class

    Returns
    -------
    g : float
         The policy's gain
    h : np.array
         The policy's differential value function if `return_bias` is set to True.
    """
    transition_matrix = get_transition_matrix(pi, mdp)

    if check_nb_irreducible_class:
        # the policy should have only one irreducible class
        # i.e only one eigen-value equal to 1, all the others with modulus <1
        tol = 1e-6
        eigval = eigvals(transition_matrix)
        multiplicity_eigenvalue_1 = len(eigval[np.where(eigval >= 1 - tol)[0]])
        assert multiplicity_eigenvalue_1 == 1

    # solves the under-constrained linear system rho•e + h = r + Ph
    # where rho is the average gain, h the bias, r the reward vector and P the transition matrix
    # here we use the pseudo-inverse to compute the solution with minimal norm
    # the under-determined system writes:
    # [e, Id-P][rho \\ h] = r rewritten below as Ax = y
    y = get_reward_vector(pi, mdp)
    mat_a = np.array([np.hstack([1, row]) for row in np.eye(mdp.n_states) - transition_matrix])
    x = np.linalg.lstsq(mat_a, y, rcond=None)[0]

    gain = x[0]

    if return_bias:
        return gain, x[1:]
    return gain
