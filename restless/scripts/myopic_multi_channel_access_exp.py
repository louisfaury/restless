"""
A simple experiment in a MultiChannelAccess scenario
Myopic vs. BestStationaryArm
"""
import matplotlib.pyplot as plt

from restless.agents import BestStationaryArmAgent, Myopic
from restless.envs import ChannelAccessMAB
from restless.bandit_game import run_exp


def main() -> None:
    n_arms = 10
    horizon = 300
    p, q = 0.8, 0.4

    # Environment
    env = ChannelAccessMAB(n_arms, p, q)

    # Agents
    stationary_agent = BestStationaryArmAgent(n_arms, [arm.stationary_reward() for arm in env.arm_list])
    myopic_agent = Myopic(
        n_arms, [arm.transition_matrix for arm in env.arm_list], [arm.reward_vector for arm in env.arm_list]
    )

    # # Compare mean rewards
    stationary_report = run_exp(stationary_agent, env, horizon)
    print(f"StationaryAgent mean reward is {stationary_report['reward'].mean()}")

    myopic_report = run_exp(myopic_agent, env, horizon)
    print(f"Myopic mean reward is {myopic_report['reward'].mean()}")

    # Display belief evolution for myopic
    plt.figure(figsize=(12, 8))
    beliefs = myopic_report["belief"].values

    for arm in range(n_arms):
        plt.plot([belief[arm] for belief in beliefs], label=f"arm {arm}")
    stationary_distribution = env.arm_list[0].stationary_distribution()[1]
    plt.plot([0, horizon], [stationary_distribution, stationary_distribution], label="stationary")
    plt.xlabel("Round")
    plt.ylabel("Belief")
    plt.title("Evolution of belief's under a myopic strategy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
