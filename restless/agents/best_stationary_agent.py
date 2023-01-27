"""
Agent that always plays the arm with highest stationary reward
"""

from typing import List, Union

import numpy as np
from restless.agents.agent import Agent


class BestStationaryArmAgent(Agent):
    """
    Best Stationary Arm agent
    Has knowledge about each arm's stationary reward
    """

    def __init__(self, n_arms: int, stationary_rewards: Union[List[float], np.array]):
        super().__init__(n_arms)
        self.best_stationary_arm = int(np.argmax(stationary_rewards))

    def act(self) -> int:
        return self.best_stationary_arm

    def update(self, arm: int, state: int) -> None:
        pass
