"""
Myopic agent
"""

from typing import List

import numpy as np

from restless.agents.agent import Agent


class Myopic(Agent):
    """
    Myopic agent class.
    This agent is given the true transition probabilities of the arms
    """

    def __init__(
        self,
        n_arms: int,
        transition_matrix_list: List[np.array],
        reward_vector_list: List[np.array],
    ):
        super().__init__(n_arms)

        self.transition_matrix_list = transition_matrix_list
        self.reward_vector_list = reward_vector_list

        # belief vector
        self.belief_matrix = [
            np.zeros(self.transition_matrix_list[arm].shape[0]) for arm in range(self.n_arms)
        ]

        # # sufficient statistics to compute the belief vector
        # last observed states for each arms
        self.last_observed_state = np.zeros(self.n_arms)
        # time since last observation
        self.time_since_last_observed = np.zeros(self.n_arms)

    def act(self) -> int:
        """
        The myopic policy plays the arm which belief reward is highest
        """
        if np.any(self.time_since_last_observed == 0):
            return np.where(self.time_since_last_observed == 0)[0][0]

        return int(
            np.argmax(
                [
                    np.sum(belief * reward_vector)
                    for belief, reward_vector in zip(self.belief_matrix, self.reward_vector_list)
                ]
            )
        )

    def update(self, arm: int, state: int) -> None:
        """
        Update the belief vector
        """
        # Updating first the sufficient statistics
        self.last_observed_state[arm] = state
        self.time_since_last_observed[arm] = 0
        for arm_ in range(self.n_arms):
            self.time_since_last_observed[arm_] += 1

        # Updating the belief vector for arms that were not sense
        for arm_ in range(self.n_arms):
            if arm_ == arm:
                continue
            old_belief = self.belief_matrix[arm_]
            transition_matrix = self.transition_matrix_list[arm_]
            new_belief = (old_belief.T @ transition_matrix).T
            self.belief_matrix[arm_] = new_belief

        # Updating the belief for the arm that was sense
        state_space_size = self.transition_matrix_list[arm].shape[0]
        state_vector = np.array([s == state for s in range(state_space_size)]).astype(int)
        new_belief = (state_vector.T @ self.transition_matrix_list[arm]).T
        self.belief_matrix[arm] = new_belief
