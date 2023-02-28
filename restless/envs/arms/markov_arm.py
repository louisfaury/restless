"""
Markovian arm structure
"""

from typing import Tuple

import numpy as np
import scipy.linalg


class MarkovArm:
    """
    Simple Markovian Arm (e.g. reward is a Markov chain)
    """

    def __init__(
        self,
        n_states: int,
        transition_matrix: np.array,
        reward_vector: np.array,
        init_state: int = 0,
    ):
        """
        Parameters
        ----------
        n_states : int
            Number of state in the Markov Chain
        transition_matrix : np.array
            MC's transition kernel, must be a stochastic matrix (will raise an error if not)
        reward_vector : np.array
            Defines the rewards associated with each state
        init_state : int
            Initial state
        """
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

    def stationary_distribution(self) -> np.array:
        """
        Compute the Markov chain's stationary distribution

        .. warning:: We assume existence and uniqueness of the stationary distribution
            i.e. the chain is ergodic (the transition matrix has left eigen-value 1 with multiplicity 1)
        :return: the MC's stationary distribution's vector
        :rtype: np.array
        TODO: implement further check for MC ergodicity
        TODO: the current checks are a bit sketchy (or downright false) -- eigen-values are complex numbers!
        """
        eig_values, eig_vector = scipy.linalg.eig(
            self.transition_matrix, left=True, right=False
        )  # left eigen-vectors

        # order according to descending eig_values
        order_permutation = np.argsort(eig_values)

        max_eig_values = eig_values[order_permutation[-1]]
        # the eigvector associated with highest eigvalue is (almost) the stationary distribution
        max_eig_vector = eig_vector[:, order_permutation[-1]]

        # simple check for ergodicity
        assert max_eig_values == 1
        assert eig_values[order_permutation[-2]] != 1
        assert len(np.unique([np.sign(e) for e in max_eig_vector])) == 1

        # de-normalized eigen-vector (stationary distribution sums to 1)
        # turn positive if needed
        stationary_distribution = np.abs(max_eig_vector / np.sum(max_eig_vector))
        return stationary_distribution

    def stationary_reward(self) -> float:
        """
        Returns the arm's stationary reward
        """
        return float(self.stationary_distribution().T @ self.reward_vector)
