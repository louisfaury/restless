"""
Basic agent structure
"""

from abc import abstractmethod


class Agent:
    """
    Basic agent class
    """

    def __init__(self, n_arms: int):
        self.n_arms = n_arms

    @abstractmethod
    def act(self) -> int:
        """
        Play an arm
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, arm: int, state: int) -> None:
        """
        Incorporate signal of the arm that was pulled
        """
        raise NotImplementedError

    def report(self):
        """
        Returns internal state, for logging purposes
        """
        return {}


class StubbornAgent(Agent):
    """
    Agent that constantly plays the same action (for debugging purpose)
    """

    def act(self) -> int:
        return 0

    def update(self, arm: int, state: int) -> None:
        pass
