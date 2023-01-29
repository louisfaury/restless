"""
Some standard MAB environments
"""

from typing import List, Tuple

from restless.envs.arms.markov_arm import MarkovArm
from restless.envs.arms.channel import Channel


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

    def reset_state(self, state: List[int]) -> None:
        """
        Resets all the arm state
        """
        for i, arm in enumerate(self.arm_list):
            arm.state = state[i]


class ChannelAccessMAB(RestlessMAB):  # pylint: disable=too-few-public-methods
    """
    A ChannelAccessMAB has K independent Channel arms, each identical (same p, q)
    """

    def __init__(self, n_arms: int, p: float, q: float):
        self.p = p
        self.q = q
        super().__init__([Channel(p, q) for _ in range(n_arms)])
