"""
Policy structure
TODO: test myopic
"""

from typing import List

import numpy as np
from restless.control.mdp import MarkovDecisionProcess as MDP


class Policy:
    """
    For simplicity (and because even POMDPs will be treated as extended MDPs),
    a policy must be stationary, Markovian and deterministic.
    i.e. no randomized, history-dependent, non-stationary policies
    """

    def __init__(self, actions: List[int]):
        self.actions = actions


class MyopicPolicy(Policy):
    """
    A myopic policy (optimize for one-step reward)
    """

    def __init__(self, mdp: MDP):
        all_rewards = np.array(mdp.reward_function).reshape(mdp.n_actions, mdp.n_states)
        myopic_actions = list(np.argmax(all_rewards, axis=0))
        super().__init__(myopic_actions)
