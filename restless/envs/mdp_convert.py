"""
Functions to a RestlessMAB environment to a MDP
For now, only implement for the ChannelAccessMAB model
TODO test big time
TODO check size with all permutations of size k
"""
from typing import Tuple, Dict, Callable, List

import itertools
import numpy as np
from restless.envs import ChannelAccessMAB, RestlessMAB
from restless.control import MDP


def convert_to_mdp(restless: RestlessMAB, truncate: int = 10) -> MDP:
    """
    Returns the belief-MDP, truncated after :truncate: steps in the countable belief space
    """
    if isinstance(restless, ChannelAccessMAB):
        return convert_channel_to_mdp(restless, truncate)
    else:
        raise NotImplementedError


def convert_channel_to_mdp(chanel_mab: ChannelAccessMAB, truncate: int = 10):
    """
    Belief MDP for a ChannelAccessMAB (easier and less expensive than general case)
    """
    p = chanel_mab.p
    q = chanel_mab.q

    # # Preliminaries
    # we will need the one-step belief-transition function
    def single_belief_transition(belief: float) -> float:
        return q + belief * (p - q)

    # and its n-steps variant
    def step_belief_transition_fn(n_step: int) -> Callable[[float], float]:
        def n_step_belief_transition(belief: float) -> float:
            for _ in range(n_step):
                belief = single_belief_transition(belief)
            return belief

        return n_step_belief_transition

    # # Generates all belief and assign them and index.
    per_arm_belief = [step_belief_transition_fn(n_step=k)(p) for k in range(truncate)] + [
        step_belief_transition_fn(n_step=k)(q) for k in range(truncate)
    ]
    # beliefs are sorted by decreasing value -> (p, T(p), .., (T^10)(q), ..q) if p>q
    per_arm_belief = np.sort(per_arm_belief)[::-1].tolist()

    # propagate to all the arms
    # by generating all permutations of size n_arms
    all_beliefs = [list(x) for x in itertools.product(per_arm_belief, repeat=len(chanel_mab.arm_list))]
    idx_to_belief = dict(enumerate(all_beliefs))
    belief_to_idx = {tuple(belief): k for k, belief in enumerate(all_beliefs)}

    # # MDP construction
    n_states = len(all_beliefs)
    n_actions = len(chanel_mab.arm_list)

    # # Construct the transition kernels and reward function
    # Reward function
    reward_function = [np.array([idx_to_belief[s][a] for s in range(n_states)]) for a in range(n_actions)]
    # Transition matrix

    # truncated dynamics transition
    def truncated_single_belief_transition(belief: float) -> float:
        next_belief = single_belief_transition(belief)
        return next_belief if next_belief in per_arm_belief else belief

    transition_matrix = []
    for action in range(n_actions):
        transition_kernel = np.zeros((n_states, n_states))
        for state in range(n_states):
            belief = idx_to_belief[state]

            next_belief_if_0 = [
                truncated_single_belief_transition(belief[a]) if a != action else q for a in range(n_actions)
            ]
            next_idx_if_0 = belief_to_idx[tuple(next_belief_if_0)]
            transition_kernel[state, next_idx_if_0] = 1 - belief[action]

            next_belief_if_1 = [
                truncated_single_belief_transition(belief[a]) if a != action else p for a in range(n_actions)
            ]
            next_idx_if_1 = belief_to_idx[tuple(next_belief_if_1)]
            transition_kernel[state, next_idx_if_1] = belief[action]
        transition_matrix.append(transition_kernel)

    return MDP(n_states, n_actions, transition_matrix, reward_function)
