"""
usage: myopic_multi_channel_access_exp.py [-h] [-n N] [-p P] [-q Q] [-hz HZ]

Myopic vs. BestStationaryArm in a MultiChannelAccess scenario

optional arguments:
  -h, --help  show this help message and exit
  -n N        Number of arms (default: None)
  -p P        Proba p (staying in 1) (default: None)
  -q Q        Proba q (transitioning to 1) (default: None)
  -hz HZ      Horizon (default: 1000)
"""
import argparse
import matplotlib.pyplot as plt

from restless.agents import BestStationaryArmAgent, Myopic
from restless.envs import ChannelAccessMAB
from restless.bandit_game import run_exp


def main(n_arms: int, p: float, q: float, horizon: int) -> None:
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
    plt.title("Evolution of beliefs under a myopic strategy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Myopic vs. BestStationaryArm in a MultiChannelAccess scenario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", type=int, help="Number of arms")
    parser.add_argument("-p", type=float, help="Proba p (staying in 1)")
    parser.add_argument("-q", type=float, help="Proba q (transitioning to 1)")
    parser.add_argument("-hz", type=int, default=100, help="Horizon")
    args = parser.parse_args()
    main(args.n, args.p, args.q, args.hz)
