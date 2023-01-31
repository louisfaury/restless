"""
usage: myopic_policy_evaluation_exp.py [-h] [-n N] [-p P] [-q Q] [-t T]

Policy evaluation with myopic policy in ChannelAccess env

optional arguments:
  -h, --help  show this help message and exit
  -n N        Number of arms (default: None)
  -p P        Proba p (staying in 1) (default: None)
  -q Q        Proba q (transitioning to 1) (default: None)
  -t T        Truncation steps (default: 10)
"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


from restless.control import (
    MyopicPolicy,
    DiscountedMDP,
    discounted_policy_evaluation,
    average_reward_policy_evaluation,
)
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
    # myopic policy
    pi = MyopicPolicy(channel_mdp)

    # average-gain policy evaluation
    logger.info("Computing gain and bias")

    check_nb_irreducible_class = n_arms == 2  # to win some time
    gain, bias = average_reward_policy_evaluation(pi, channel_mdp, True, check_nb_irreducible_class)

    # # Let's first check the convergence of discounted value towards the gain
    logger.info("Discounted value convergence")
    logger.info("\t Computing discounted values")
    discounts = np.array([1 - 1 / eps for eps in np.arange(10, 2_000, 100)])
    discounted_values = np.array(
        [
            discounted_policy_evaluation(pi, DiscountedMDP.from_mdp(channel_mdp, discount))
            for discount in tqdm(discounts)
        ]
    )
    logger.info("\t Plotting convergence graph")
    plt.figure(num="Discounted approach conv.", figsize=(10, 8))
    plt.plot(range(len(discounts)), gain * np.ones_like(discounts), label="Avg-gain")
    plt.plot(range(len(discounts)), (1 - discounts) * discounted_values[:, 0], label="Discounted")
    plt.xlabel("Discount")
    plt.ylabel("Value")
    plt.xticks(range(len(discounts))[1::4], np.round(discounts[1::4], 4))
    plt.legend()
    plt.show()

    # # Monotonicity of the value function (only for 2 ams)
    if n_arms == 2:
        logger.info("Differential function monotonicity (only for n_arms == 2)")
        idx_to_belief = idx_belief_maps[0]
        belief_to_idx = idx_belief_maps[1]

        all_ordered_beliefs = sorted(list(idx_to_belief.values()))

        logger.info("\t Displaying full plot (only 2 ams)")
        # let's order belief, from first coordinate to second
        data_matrix = np.array(
            [bias[belief_to_idx[tuple(belief)]] for belief in all_ordered_beliefs]
        ).reshape((2 * truncate, 2 * truncate))

        plt.figure(num="2d val", figsize=(10, 8))
        plt.contourf(data_matrix)
        plt.colorbar()
        plt.title("Differential-value function")
        plt.xlabel("Belief arm 0")
        plt.ylabel("Belief arm 1")
        plt.xticks([0, 2 * truncate - 1], [all_ordered_beliefs[0][0], all_ordered_beliefs[-1][0]])
        plt.yticks([0, 2 * truncate - 1], [all_ordered_beliefs[0][0], all_ordered_beliefs[-1][0]])
        plt.show()

    logger.info("\t Displaying projected differential value function")
    plt.figure(num="Projected val", figsize=(10, 8))
    # differential-value function on rays of increasing belief
    for belief_idx in np.arange(0, 4 * truncate**2, 2 * truncate):
        sorted_beliefs = all_ordered_beliefs[belief_idx : belief_idx + 2 * truncate]
        plt.plot([bias[belief_to_idx[tuple(belief)]] for belief in sorted_beliefs])

    plt.xticks([0, 2 * truncate - 1], [all_ordered_beliefs[0][0], all_ordered_beliefs[-1][0]])
    plt.xlabel("Increasing belief over 2nd arm")
    plt.ylabel("Differential value")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Policy evaluation with myopic policy in ChannelAccess env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", type=int, help="Number of arms")
    parser.add_argument("-p", type=float, help="Proba p (staying in 1)")
    parser.add_argument("-q", type=float, help="Proba q (transitioning to 1)")
    parser.add_argument("-t", type=int, default=20, help="Truncation steps")
    args = parser.parse_args()

    main(args.n, args.p, args.q, args.t)
