"""
Markovian arm structure

TODO: Compute stationary distribution
TODO: Protect state
"""

from typing import Tuple

import numpy as np


class MarkovArm:
    """
    Simple Markovian Arm (e.g. reward is a Markov chain)

    Then transition matrix must be a stochastic matrix (each row sum to 1, only positive entries)
    """

    def __init__(
        self,
        n_states: int,
        transition_matrix: np.array,
        reward_vector: np.array,
        init_state: int = 0,
    ):
        self.n_states = n_states

        assert np.shape(transition_matrix)[0] == np.shape(transition_matrix)[1]
        assert np.shape(transition_matrix)[0] == self.n_states
        assert transition_matrix.min() >= 0
        assert np.allclose(transition_matrix @ np.ones(self.n_states), np.ones(self.n_states).T)
        self.transition_matrix = transition_matrix

        assert init_state < self.n_states
        self.state = init_state

        assert len(reward_vector) == self.n_states
        self.reward_vector = np.reshape(reward_vector, (self.n_states,))

        # random generator
        self.rng = np.random.default_rng()

    @property
    def state_vector(self) -> np.array:
        """
        0-1 representation of the state
        """
        return np.array([s == self.state for s in range(self.n_states)]).astype(int)

    def step(self) -> None:
        """
        Samples the next state
        """
        transition_kernel = (self.state_vector.T @ self.transition_matrix).T
        self.state = self.rng.choice(self.n_states, p=transition_kernel)

    def reward(self) -> float:
        """
        Returns the reward associated with current state
        """
        return self.reward_vector.T @ self.state_vector

    def sense(self) -> Tuple[int, float]:
        "Returns current state and associated reward"
        return self.state, self.reward()
