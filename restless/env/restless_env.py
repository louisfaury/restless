"""
Some standard MAB environments
"""

from typing import List, Tuple

from restless.env.arms.markov_arm import MarkovArm
from restless.env.arms.channel import Channel


class RestlessMAB:  # pylint: disable=too-few-public-methods
    """
    RestlessMAB class.
    Update the different arms given an agent's action
    """

    def __init__(self, arms: List[MarkovArm]):
        self.arm_list = arms

    def sense(self, arm: int) -> Tuple[int, float]:
        """
        Returns the current state and reward for the selected arm
        Update all arms
        """
        assert arm < len(self.arm_list)
        state, reward = self.arm_list[arm].sense()

        for arm_ in self.arm_list:
            arm_.step()

        return state, reward


class ChannelAccessMAB(RestlessMAB):  # pylint: disable=too-few-public-methods
    """
    A ChannelAccessMAB has K independent Channel arms, each identical (same p, q)
    """

    def __init__(self, n_arms: int, p: float, q: float):
        super().__init__([Channel(p, q) for _ in range(n_arms)])
