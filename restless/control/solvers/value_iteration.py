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


def discounted_value_iterations(mdp: DiscountedMDP, precision: float, max_iter: int = 10_000) -> np.array:
    """
    Run the discounted value iteration algorithm to obtain the optimal policy's value
    up to :param precision: (capped at :param max_iter:).
    """
    initial_value = np.zeros((mdp.n_states,))

    def one_step_vi(cur_value: np.array):
        """
        Single step of discounted value iteration
        """
        return np.array(
            [
                np.max(
                    [
                        mdp.reward_function[action] + mdp.discount * mdp.transition_kernel[action] @ cur_value
                        for action in range(mdp.n_actions)
                    ],
                    axis=0,
                )
            ]
        ).reshape((mdp.n_states,))

    # compute the number of steps needed to reach required precision
    first_value = one_step_vi(initial_value)
    iter_ub = log(precision * (1 - mdp.discount) / np.linalg.norm(first_value - initial_value)) / log(
        mdp.discount
    )
    steps_to_precision = min([max_iter, int(iter_ub)])
    logger.debug(f"Number of iterations: {steps_to_precision}")
    # let's go
    logger.debug(f"Running VI for {steps_to_precision} steps")
    value = first_value
    for t in tqdm(range(steps_to_precision)):
        next_value = one_step_vi(value)
        # detect convergence
        if np.linalg.norm(next_value - value) <= (1 - mdp.discount) * precision / mdp.discount:
            logger.debug(f"Round {t} detected convergence: {np.linalg.norm(next_value - value)}")
            return next_value
        value = next_value

    return value


def relative_value_iteration(
    mdp: MDP, precision: float, max_iter: int = 1_000, return_bias: bool = False
) -> Union[float, Tuple[float, np.array]]:
    """
    Runs the relative VI algorithm to find the optimal (gain, bias) couple in an average-gain MDP.
    Goal is to find it up to some precision, given some maximum number of iterations.

    .. warning:: This assumes that the MDP is at least weakly-communicating. We don't check this property,
        so use at your own risk.
    :param precision: Desired accuracy level
    :type precision: float
    :param max_iter: Maximum number of iterations
    :type max_iter: int
    :param return_bias: Wether to return the associated differential value function
    :type return_bias: bool
    :return: (g, h) the estimated mdp's optimal gain, as well as its associated bias (differential value function).
    """
    # aperiodicity transform
    tau = 0.05  # default value, should not require tuning
    aperiodic_mdp = mdp.to_aperiodic_mdp(tau)

    # single step relative value iteration
    def single_step_rvi(value: np.array):
        normalized_value = value
        return np.array(
            [
                np.max(
                    [
                        aperiodic_mdp.reward_function[action]
                        + aperiodic_mdp.transition_kernel[action] @ normalized_value
                        for action in range(mdp.n_actions)
                    ],
                    axis=0,
                )
            ]
        ).reshape((mdp.n_states,))

    # many steps relative value iteration
    logger.debug(f"Running RVI for {max_iter} steps")
    value = np.zeros((mdp.n_states,))
    for t in tqdm(range(max_iter)):
        next_value = single_step_rvi(value)
        # checks convergence via span semi-norm
        span = np.max(next_value - value) - np.min(next_value - value)
        if span < precision:  # span semi-norm convergence
            logger.debug(f"Detected convergence at step {t}")
            break

        if t + 1 < max_iter:
            value = next_value
        else:
            logger.warning(f"Relative value iteration stopped at {max_iter}. Span semi-norm is {span}.")

    # return bias and gain
    aperiodic_gain = 0.5 * (np.max(next_value - value) + np.min(next_value - value))
    gain = aperiodic_gain / tau
    if not return_bias:
        return gain

    else:
        aperiodic_bias = next_value - next_value[0]
        bias = aperiodic_bias  # / tau

        return gain, bias
