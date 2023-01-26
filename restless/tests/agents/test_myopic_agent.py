import numpy as np
from restless.agents import Myopic
from restless.envs import RestlessMAB, Channel


def test_myopic_with_dominated_arms():
    """
    A simple environment where one arm is completely dominated by the other
    Myopic should always play the best arm
    """

    # # Environment
    best_agent = Channel(0.8, 0.4)
    # manually changing the reward of worst_agent to guarantees its reward vector is dominated
    worst_agent = Channel(0.8, 0.4)
    worst_agent.reward_vector = np.array([-1, -1])

    env = RestlessMAB([best_agent, worst_agent])

    # # agent
    agent = Myopic(
        n_arms=2,
        transition_matrix_list=[arm.transition_matrix for arm in env.arm_list],
        reward_vector_list=[arm.reward_vector for arm in env.arm_list],
    )

    # # make sure the agent has seen realizations to update its beliefs
    for arm, _ in enumerate(env.arm_list):
        state, _ = env.sense(arm)
        agent.update(arm, state)

    # # run experiment
    horizon = 100
    arm_play = np.empty(horizon)

    for t in range(horizon):
        action = agent.act()
        state, _ = env.sense(action)
        agent.update(action, state)
        arm_play[t] = action

    np.testing.assert_equal(arm_play, np.zeros(horizon))
