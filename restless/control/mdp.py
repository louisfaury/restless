"""
MDP structure
"""
from typing import List

import numpy as np


class MarkovDecisionProcess:
    """
    Finite state and action space MDP class.
    Transition and rewards are assumed stationary.
    Default is to consider average-cost objective.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        transition_kernel: List[np.array],
        reward_function: List[np.array],
    ):
        """
        Then transition kernel must be a list containing the transition matrices for each action
        The reward function is a list containing the reward vector for each action
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_kernel = transition_kernel
        self.reward_function = reward_function

        # # some basic sanity checks
        assert len(transition_kernel) == n_actions
        assert len(reward_function) == n_actions
        assert np.all(
            [self.transition_kernel[k].shape == (self.n_states, self.n_states) for k in range(self.n_actions)]
        )
        assert np.all([self.reward_function[k].shape == (self.n_states,) for k in range(self.n_actions)])
        # all transition matrix are stochastic
        assert np.allclose(
            [self.transition_kernel[k].sum(axis=1) for k in range(self.n_actions)],
            np.ones((self.n_actions, self.n_states)),
        )
        assert np.all([self.transition_kernel[k].min() >= 0 for k in range(self.n_actions)])

    def to_aperiodic_mdp(self, tau: float) -> "MarkovDecisionProcess":
        """
        Aperiodicity transform of an MDP (see Puterman §8.5.4)
        """
        assert (tau > 0) & (tau < 1)

        # change reward function
        new_reward_function = [tau * reward_vector for reward_vector in self.reward_function]
        # change transition kernel
        new_transition_kernel = [
            (1 - tau) * np.eye(self.n_states) + tau * transition_matrix
            for transition_matrix in self.transition_kernel
        ]

        return MarkovDecisionProcess(
            self.n_states, self.n_actions, new_transition_kernel, new_reward_function
        )


class DiscountedMarkovDecisionProcess(MarkovDecisionProcess):
    """
    With discount
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        transition_kernel: List[np.array],
        reward_function: List[np.array],
        discount: float,
    ):
        super().__init__(n_states, n_actions, transition_kernel, reward_function)
        self.discount = discount
        assert (self.discount >= 0) & (self.discount < 1)

    @classmethod
    def from_mdp(cls, mdp: MarkovDecisionProcess, discount):  # pragma: no cover
        return cls(mdp.n_states, mdp.n_actions, mdp.transition_kernel, mdp.reward_function, discount)


class FiniteHorizonMarkovDecisionProcess(MarkovDecisionProcess):
    """
    With discount
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        transition_kernel: List[np.array],
        reward_function: List[np.array],
        horizon: int,
    ):
        super().__init__(n_states, n_actions, transition_kernel, reward_function)
        self.horizon = horizon
