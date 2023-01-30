"""
Helper to easily run a Restless MAB game and log relevant history.
"""
from typing import List

import pandas as pd

from restless.agents import Agent
from restless.envs import RestlessMAB


def run_exp(
    agent: Agent, env: RestlessMAB, horizon: int, verbose: bool = False
) -> pd.DataFrame:  # pragma: no cover
    """
    Returns the total reward accumulated by an agent when playing for horizon time-steps.
    """
    print(f"Start experiment with {len(env.arm_list)} arms and {type(agent).__name__} for {horizon} rounds")

    # by default all arm start in same state # TODO allow different initial states
    env.reset_state([0 for _ in range(len(env.arm_list))])

    report: List[dict] = [{} for _ in range(horizon)]  # list of empty dict
    # let's go
    for t in range(horizon):
        arm = agent.act()
        state, reward = env.sense(arm)
        agent.update(arm, state)

        if verbose:
            print(f"Agent played arm {arm}, observed state {state} and received reward {reward}")

        # logging
        report[t]["arm"] = arm
        report[t]["reward"] = reward
        report[t]["states"] = [arm.state for arm in env.arm_list]

        agent_report = agent.report()
        for key in agent_report.keys():
            report[t][key] = agent_report[key]

    return pd.DataFrame(report)
