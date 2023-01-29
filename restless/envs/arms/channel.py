"""
Channel arm structure
"""


import numpy as np

from restless.envs.arms.markov_arm import MarkovArm


class Channel(MarkovArm):
    """
    Markov arm with two states.
    1 has reward 1, 0 has reward 0.
    """

    def __init__(self, p: float, q: float, init_state: int = 1):
        """
        p: probability of staying in 1 starting from 1
        q: probability of going to 1 starting from 0
        """
        transition_matrix = np.array([[1 - q, q], [1 - p, p]])
        reward_vector = np.array([0, 1])
        super().__init__(
            n_states=2,
            transition_matrix=transition_matrix,
            reward_vector=reward_vector,
            init_state=init_state,
        )
        self.p = p
        self.q = q

    def stationary_distribution(self):
        """
        In this case we have a closed form for the stationary distribution
        """
        p = self.p
        q = self.q

        return np.array([1 - q / (1 + q - p), q / (1 + q - p)])
