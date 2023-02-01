"""
usage: optimal_policy_inspection_exp.py [-h] [-n N] [-p P] [-q Q] [-t T]

Computes and inspects the optimal policy in a ChannelAccessMAB.

optional arguments:
  -h, --help  show this help message and exit
  -n N        Number of arms (default: None)
  -p P        Proba p (staying in 1) (default: None)
  -q Q        Proba q (transitioning to 1) (default: None)
  -t T        Truncation steps (default: 20)
"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np


from restless.control import MyopicPolicy, relative_value_iteration, average_reward_policy_evaluation
from restless.envs import ChannelAccessMAB
from restless.envs.mdp_convert import convert_channel_to_mdp


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(n_arms, p, q, truncate):
    logger.info(f"Average-MDP experiments with the myopic policy, n_arms={n_arms}, p={p}, q={q}")

    logger.info("Creating env and agent")
    # defining environment
    channel_mab = ChannelAccessMAB(n_arms, p, q)

    # converting to MDP
    channel_mdp, idx_belief_maps = convert_channel_to_mdp(
        channel_mab, truncate=truncate, return_belief_idx=True
    )

    # average-gain policy evaluation
    logger.info("Computing optimal gain and differential value function")
    optimal_gain, optimal_bias = relative_value_iteration(
        channel_mdp, max_iter=10_000, precision=1e-5, return_bias=True
    )
    # TODO: improving policy of optimal gain
    logger.info(f"Optimal average-gain is rho={optimal_gain}")

    # myopic policy
    pi = MyopicPolicy(channel_mdp)
    myopic_gain, _ = average_reward_policy_evaluation(
        pi, channel_mdp, True, check_nb_irreducible_class=False
    )
    logger.info(f"Myopic average-gain is rho={myopic_gain}")

    if np.abs(optimal_gain - myopic_gain) < 1e-4:
        logger.info("\033[1m Myopic is optimal!\033[0m")

    # # Monotonicity of optimal the value function (only for 2 ams)
    if n_arms == 2:
        logger.info("Differential function monotonicity (only for n_arms == 2)")
        idx_to_belief = idx_belief_maps[0]
        belief_to_idx = idx_belief_maps[1]

        all_ordered_beliefs = sorted(list(idx_to_belief.values()))

        logger.info("\t Displaying full plot (only 2 ams)")
        # let's order belief, from first coordinate to second
        data_matrix = np.array(
            [optimal_bias[belief_to_idx[tuple(belief)]] for belief in all_ordered_beliefs]
        ).reshape((2 * truncate, 2 * truncate))

        plt.figure(num="2d val", figsize=(10, 8))
        plt.contourf(data_matrix)
        plt.colorbar()
        plt.title("Optimal differential-value function")
        plt.xlabel("Belief arm 0")
        plt.ylabel("Belief arm 1")
        plt.xticks([0, 2 * truncate - 1], [all_ordered_beliefs[0][0], all_ordered_beliefs[-1][0]])
        plt.yticks([0, 2 * truncate - 1], [all_ordered_beliefs[0][0], all_ordered_beliefs[-1][0]])
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Computes and inspects the optimal policy in a ChannelAccessMAB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", type=int, help="Number of arms")
    parser.add_argument("-p", type=float, help="Proba p (staying in 1)")
    parser.add_argument("-q", type=float, help="Proba q (transitioning to 1)")
    parser.add_argument("-t", type=int, default=20, help="Truncation steps")
    args = parser.parse_args()

    main(args.n, args.p, args.q, args.t)
