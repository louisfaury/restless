"""
Value iteration for discounted and average-gain MDPs
"""

import logging
from typing import Union, Tuple

import numpy as np

from math import log
from tqdm import tqdm
from restless.control.mdp import MarkovDecisionProcess as MDP
from restless.control.mdp import DiscountedMarkovDecisionProcess as DiscountedMDP


logger = logging.getLogger(__name__)


def one_step_value_iteration(mdp: MDP, value: np.array):
    """
    One step of Value Iteration

    Parameters
    ----------
    mdp : MarkovDecisionProcess
    value : np.array
        The initial value function

    Returns
    -------
    : np.array
        The resulting value function after one step of Value Iteration
    """
    discount = mdp.discount if isinstance(mdp, DiscountedMDP) else 1

    return np.array(
        [
            np.max(
                [
                    mdp.reward_function[action] + discount * mdp.transition_kernel[action] @ value
                    for action in range(mdp.n_actions)
                ],
                axis=0,
            )
        ]
    ).reshape((mdp.n_states,))


def discounted_value_iterations(mdp: DiscountedMDP, precision: float, max_iter: int = 10_000) -> np.array:
    """
    Run the discounted value iteration algorithm to obtain the optimal policy's value

    Parameters
    ----------
    mdp : DiscountedMarkovDecisionProcess
    precision : float
        Required ell_infinity error for the algorithm to stop
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    : np.array
        The array representing the optimal policy's value function
    """
    initial_value = np.zeros((mdp.n_states,))

    # compute the number of steps needed to reach required precision
    first_value = one_step_value_iteration(mdp, initial_value)
    iter_ub = log(precision * (1 - mdp.discount) / np.linalg.norm(first_value - initial_value)) / log(
        mdp.discount
    )
    steps_to_precision = min([max_iter, int(iter_ub)])
    logger.debug(f"Number of iterations: {steps_to_precision}")

    # let's go
    logger.debug(f"Running VI for {steps_to_precision} steps")
    value = first_value
    for t in tqdm(range(steps_to_precision)):
        next_value = one_step_value_iteration(mdp, value)
        # stops if convergence is detected earlier than anticipated
        if np.linalg.norm(next_value - value) <= (1 - mdp.discount) * precision / mdp.discount:
            logger.debug(f"Round {t} detected convergence: {np.linalg.norm(next_value - value)}")
            return next_value
        value = next_value

    return value


def value_iteration(
    mdp: MDP, precision: float, max_iter: int = 1_000, return_bias: bool = False
) -> Union[float, Tuple[float, np.array]]:
    """
    Runs the VI algorithm to find the optimal (gain, bias) couple in an average-gain MDP.
    The goal is to find it up to some precision, given some maximum number of iterations.

    .. warning:: This assumes that the MDP is at least weakly-communicating. We don't check this property,
        so use at your own risk.

    Parameters
    ----------
    mdp : MarkovDecisionProcess
    precision : float
        Desired accuracy level
    max_iter : int
        Maximum number of iterations
    return_bias: Optional[bool]
        Wether to return the associated differential value function

    Returns
    -------
    g : float
        The MDP (estimated) optimal gain
    h : np.array
        The MDP (estimated) bias (or differential value function)
    """
    # Aperiodicity transform
    tau = 0.5  # default value, does not require tuning
    aperiodic_mdp = mdp.to_aperiodic_mdp(tau)

    # Let's go
    logger.debug(f"Running RVI for {max_iter} steps")

    value = np.zeros((mdp.n_states,))
    for t in tqdm(range(max_iter)):
        next_value = one_step_value_iteration(aperiodic_mdp, value)
        # checks convergence via span semi-norm
        span = np.max(next_value - value) - np.min(next_value - value)
        if span < precision:
            logger.debug(f"Detected convergence at step {t}")
            break

        if t + 1 < max_iter:
            value = next_value
        else:
            logger.warning(f"Relative value iteration stopped at {max_iter}. Span semi-norm is {span}.")

    # Computing gain andd differential value value
    aperiodic_gain = 0.5 * (np.max(next_value - value) + np.min(next_value - value))
    gain = aperiodic_gain / tau
    if not return_bias:
        return gain
    else:
        aperiodic_bias = next_value - next_value[0]
        bias = aperiodic_bias

        return gain, bias
