
import numpy as np
from restless.agents import BestStationaryArmAgent
from restless.envs import RestlessMAB, Channel


def test_stationary_with_dominated_arms():
    """
    A simple environment where one arm's stationary reward dominates the other
    BestStationaryArmAgent should always play said arm
    """

    # # Environment (best/worst w.r.t to stationary reward)
    best_agent = Channel(0.8, 0.4)
    worst_agent = Channel(0.2, 0.1)

    env = RestlessMAB([best_agent, worst_agent])

    # # agent
    agent = BestStationaryArmAgent(
        n_arms=2,
        stationary_rewards=[arm.stationary_reward() for arm in env.arm_list]
    )

    # # run experiment
    horizon = 100
    arm_play = np.empty(horizon)

    for t in range(horizon):
        action = agent.act()
        state, _ = env.sense(action)
        agent.update(action, state)
        arm_play[t] = action

    np.testing.assert_equal(arm_play, np.zeros(horizon))

