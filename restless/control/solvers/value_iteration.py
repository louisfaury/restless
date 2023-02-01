"""
Value iteration for discounted and average-gain MDPs
"""

import logging
import numpy as np

from math import log
from tqdm import tqdm
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
