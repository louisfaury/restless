"""
A simple experiment in a MultiChannelAccess scenario
Myopic vs. BestStationaryArm

TODO: easy logging
TODO: visualisation
"""

import numpy as np

from restless.agents import Agent, BestStationaryArmAgent, Myopic
from restless.envs import RestlessMAB, ChannelAccessMAB


def main() -> None:
    n_arms = 50
    horizon = 10_000
    p, q = 0.9, 0.1

    # Environment
    env = ChannelAccessMAB(n_arms, p, q)

    # Agents
    stationary_agent = BestStationaryArmAgent(n_arms, [arm.stationary_reward() for arm in env.arm_list])
    myopic_agent = Myopic(
        n_arms, [arm.transition_matrix for arm in env.arm_list], [arm.reward_vector for arm in env.arm_list]
    )

    stationary_reward = run_exp(stationary_agent, env, horizon)
    myopic_reward = run_exp(myopic_agent, env, horizon)
    print(stationary_reward / horizon, myopic_reward / horizon)


def run_exp(agent: Agent, env: RestlessMAB, horizon: int) -> float:
    """
    Returns the total reward accumulated by an agent when playing for horizon time-steps.
    """
    env.reset_state([0 for _ in range(len(env.arm_list))])
    total_reward = 0.0
    for _ in range(horizon):
        arm = agent.act()
        if len(np.array([arm])) > 1:
            raise ValueError(f"{arm}")
        state, reward = env.sense(arm)
        agent.update(state, arm)
        total_reward += reward
    return float(total_reward)


if __name__ == "__main__":
    main()
