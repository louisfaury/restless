"""
Some standard MAB environments
"""

import numpy as np
from typing import List, Tuple

from restless.envs.arms.markov_arm import MarkovArm
from restless.envs.arms.channel import Channel


class RestlessMAB:  # pylint: disable=too-few-public-methods
    """
    RestlessMAB class.

    Attributes
    ----------
    arm_list : List[MarkovArms]
        A list of independent Markov arms
    """

    def __init__(self, arms: List[MarkovArm]):
        self.arm_list = arms

    def sense(self, arm: int) -> Tuple[int, float]:
        """
        Observe the current state and reward for the selected arm and update all arms.

        Parameters
        ----------
        arm : int
            The pulled arm

        Returns
        -------
        state : int
            The observed state
        reward : float
            The observed reward
        """
        assert arm < len(self.arm_list)
        state, reward = self.arm_list[arm].sense()

        for arm_ in self.arm_list:
            arm_.step()

        return state, reward

    def reset_state(self, state: List[int]) -> None:
        """
        Resets all the arm state
        """
        for i, arm in enumerate(self.arm_list):
            arm.state = state[i]


class ChannelAccessMAB(RestlessMAB):  # pylint: disable=too-few-public-methods
    """
    An homogeneous ChannelAccessMAB has K independent Channel arms, each identical (same p, q)

    Attributes
    ----------
    p : float
        Probability of staying in state 1, common to all arms
    q : float
        Probability of staying of switching form 0 to 1, common to all arms
    """

    def __init__(self, n_arms: int, p: float, q: float):
        self.p = p
        self.q = q
        super().__init__([Channel(p, q) for _ in range(n_arms)])


class HeterogeneousChannelAccessMAB(RestlessMAB):  # pylint: disable=too-few-public-methods
    """
    An heterogeneous ChannelAccessMAB (that is, with different (p, q) probabilities)

    Attributes
    ----------
    p_list : List[float]
        List of proba. to stay in state 1, one for each arm.
    q_list : List[float]
        List of proba. to switch from state 0 to state 1, one for each arm.
    """

    def __init__(self, p_list: List[float], q_list: List[float]):
        assert len(p_list) == len(q_list)

        self.p_list = p_list
        self.q_list = q_list
        super().__init__([Channel(p, q) for (p, q) in zip(self.p_list, self.q_list)])


class SymmetricRestlessMAB(RestlessMAB):  # pylint: disable=too-few-public-methods
    """
    A symmetric environment -- each arm is independent, but share the same transition matrix and
    reward vector.
    (If each arm possesses only two states, this boils down to a ChannelAccessMAB)

    Attributes
    ----------
    transition_matrix : np.array
        The transition matrix common to all arms
    reward_vector : np.array
        The reward vector common to all arms
    """

    def __init__(self, n_arms: int, transition_matrix: np.array, reward_vector: np.array):
        """

        Parameters
        ----------
        n_arms : int
            The total number of arms
        transition_matrix : np.array
             The transition matrix common to all arms
        reward_vector : np.array
            The reward vector common to all arms
        """
        super().__init__(
            [MarkovArm(len(reward_vector), transition_matrix, reward_vector) for _ in range(n_arms)]
        )
