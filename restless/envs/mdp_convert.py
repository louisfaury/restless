"""
Functions to a RestlessMAB environment to a MDP
For now, only implement for the ChannelAccessMAB model
"""
from typing import Callable, Union, Tuple, Dict
from functools import lru_cache

import itertools
import logging
import numpy as np
from restless.envs import ChannelAccessMAB, RestlessMAB
from restless.control import MDP


logger = logging.getLogger(__name__)


def convert_to_mdp(
    restless_mab: RestlessMAB, truncate: int = 10, return_belief_idx: bool = False
) -> Union[MDP, Tuple[MDP, Tuple[Dict, Dict]]]:
    """
    Belief-MDP for a restless MAB environment.
    The belief-MDP must be finite; it is therefore truncated.

    Parameters
    ----------
    restless_mab : RestlessMAB
    truncate : int
        The belief truncation size
    return_belief_idx : bool
        Whether to return the map from belief (actual value) to their index in the finite (truncated) belief-MDP

    Returns
    -------
     mdp : MarkovDecisionProcess
        The (truncated) belief MDP
    belief_idx_map: Tuple(Dict, Dict)
        The (idx -> belief) and (belief -> idx) maps
    """
    if isinstance(restless_mab, ChannelAccessMAB):
        # call the simpler convert_channel_to_mdp function
        return convert_channel_to_mdp(restless_mab, truncate, return_belief_idx)
    else:
        # call the generic converter
        return convert_generic_rmab_to_mdp(restless_mab, truncate, return_belief_idx)


def convert_generic_rmab_to_mdp(
    restless_mab: RestlessMAB, truncate: int = 10, return_belief_idx: bool = False
) -> Union[MDP, Tuple[MDP, Tuple[Dict, Dict]]]:
    """
    Belief-MDP for a generic RestlessMAB environment (i.e. not a ChannelAccessMAB).
    """
    logger.info(f"Converting ChannelMAB access with truncate={truncate}")

    # # Preliminaries
    # we will need the belief transition function, for each arm
    def step_belief_transition_fn(arm: int) -> Callable[[int], Callable[[int], np.array]]:
        """
        Returns a function that computes the belief of an arm,
        given that it is not sensed for n steps and starts in a provided state.
        """
        markov_arm = restless_mab.arm_list[arm]
        transition_matrix = markov_arm.transition_matrix

        def n_steps_belief_transition_fn(state: int) -> Callable[[int], np.array]:
            # one hot encoding of the state
            one_hot_state = np.array([idx == state for idx in range(markov_arm.n_states)]).astype(float)

            @lru_cache(maxsize=truncate)
            def n_steps_belief_transition(n: int) -> np.array:
                return one_hot_state if n == 0 else transition_matrix.T @ n_steps_belief_transition(n - 1)

            return n_steps_belief_transition

        return n_steps_belief_transition_fn

    # # Compute the truncated belief-space (compute admissible beliefs and assign them an index)
    # start by computing the admissible (truncated) beliefs for each arm
    admissible_arm_beliefs = [
        [
            step_belief_transition_fn(arm)(mc_state)(n)
            for mc_state, n in itertools.product(
                range(restless_mab.arm_list[arm].n_states), np.arange(1, truncate + 1)
            )
        ]
        for arm in range(len(restless_mab.arm_list))
    ]

    # computes the entire combination of beliefs, across all arms
    admissible_beliefs = [list(belief) for belief in itertools.product(*admissible_arm_beliefs)]
    logger.debug(f"Associated MDP has {len(admissible_beliefs)} states.")
    # and assign indexes
    idx_to_belief = dict(enumerate(admissible_beliefs))
    belief_to_idx = {tuple(map(tuple, belief)): k for k, belief in idx_to_belief.items()}

    # # MDP construction
    n_states = len(admissible_beliefs)
    n_actions = len(restless_mab.arm_list)

    # Compute the reward function
    reward_function = [
        np.array(
            [
                np.dot(restless_mab.arm_list[a].reward_vector, admissible_beliefs[s][a])
                for s in range(n_states)
            ]
        )
        for a in range(n_actions)
    ]

    # Compute the transition matrix
    transition_kernel = [np.eye(n_states) for _ in range(n_actions)]  # TODO remember to change

    # we will need the truncated one-step update
    def truncated_transition_fn(arm: int):
        def truncated_transition(arm_belief: np.array):
            next_arm_belief = restless_mab.arm_list[arm].transition_matrix.T @ arm_belief
            return next_arm_belief if next_arm_belief in np.array(admissible_arm_beliefs[arm]) else arm_belief

        return truncated_transition

    for action in range(n_actions):
        transition_matrix = np.zeros((n_states, n_states))
        for state in range(n_states):
            # compute the admissible next beliefs, knowing current belief and provided action
            belief = idx_to_belief[state]
            next_beliefs_with_probas = [
                (
                    [
                        truncated_transition_fn(arm)(belief[arm])
                        if arm != action
                        else step_belief_transition_fn(arm)(s)(1)
                        for arm in range(n_actions)
                    ],
                    belief[action][s],
                )
                for s in range(restless_mab.arm_list[action].n_states)
            ]

            for next_belief, proba in next_beliefs_with_probas:
                next_state_idx = belief_to_idx[tuple(map(tuple, next_belief))]
                transition_matrix[state, next_state_idx] = proba

        transition_kernel[action] = transition_matrix

    mdp = MDP(n_states, n_actions, transition_kernel, reward_function)
    if return_belief_idx:
        return mdp, (idx_to_belief, belief_to_idx)
    return mdp


def convert_channel_to_mdp(
    chanel_mab: ChannelAccessMAB, truncate: int = 10, return_belief_idx: bool = False
) -> Union[MDP, Tuple[MDP, Tuple[Dict, Dict]]]:
    """
    Belief-MDP for a ChannelAccessMAB. The existence of only two states per-chain and the problem's symmetry
    allow for a somewhat easier derivation than in the general case.
    """
    logger.info(f"Converting ChannelMAB access with truncate={truncate}")
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
    logger.debug(f"Associated MDP has {len(all_beliefs)} states.")

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

    transition_kernel = []
    for action in range(n_actions):
        transition_matrix = np.zeros((n_states, n_states))
        for state in range(n_states):
            belief = idx_to_belief[state]

            next_belief_if_0 = [
                truncated_single_belief_transition(belief[a]) if a != action else q for a in range(n_actions)
            ]
            next_idx_if_0 = belief_to_idx[tuple(next_belief_if_0)]
            transition_matrix[state, next_idx_if_0] = 1 - belief[action]

            next_belief_if_1 = [
                truncated_single_belief_transition(belief[a]) if a != action else p for a in range(n_actions)
            ]
            next_idx_if_1 = belief_to_idx[tuple(next_belief_if_1)]
            transition_matrix[state, next_idx_if_1] = belief[action]
        transition_kernel.append(transition_matrix)

    mdp = MDP(n_states, n_actions, transition_kernel, reward_function)
    if return_belief_idx:
        return mdp, (idx_to_belief, belief_to_idx)
    return mdp
