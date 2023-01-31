"""
Control module
"""

from restless.control.mdp import MarkovDecisionProcess as MDP
from restless.control.mdp import DiscountedMarkovDecisionProcess as DiscountedMDP
from restless.control.mdp import FiniteHorizonMarkovDecisionProcess as FiniteHorizonMDP
from restless.control.policy import Policy, MyopicPolicy

from restless.control.solvers.policy_evaluation import (
    discounted_policy_evaluation,
    average_reward_policy_evaluation,
)
