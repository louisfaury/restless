import numpy as np
import pytest

from restless.control.solvers.value_iteration import value_iteration
from restless.envs.arms.markov_arm import MarkovArm
from restless.envs import ChannelAccessMAB, SymmetricRestlessMAB, RestlessMAB
from restless.envs.mdp_convert import convert_to_mdp


@pytest.mark.parametrize(
    "n_arms, p, q, truncate, exp_n_states, exp_n_actions",
    [
        (2, 0.8, 0.4, 1, 4, 2),
        (2, 0.8, 0.4, 2, 16, 2),
        (2, 0.8, 0.4, 10, 20**2, 2),
        (3, 0.8, 0.4, 10, 20**3, 3),
        (2, 0.4, 0.8, 2, 16, 2),
    ],
)
def test_channel_converter(n_arms, p, q, truncate, exp_n_states, exp_n_actions):
    channel_mab = ChannelAccessMAB(n_arms=n_arms, p=p, q=q)

    mdp = convert_to_mdp(channel_mab, truncate)

    assert mdp.n_states == exp_n_states
    assert mdp.n_actions == exp_n_actions

    assert np.all(
        [mdp.reward_function[k].min() == min([p, q]) for k in range(mdp.n_actions)]
    )
    assert np.all(
        [mdp.reward_function[k].max() == max([p, q]) for k in range(mdp.n_actions)]
    )

    assert np.all(
        [
            mdp.transition_kernel[k].max() == max([p, 1 - p, q, 1 - q])
            for k in range(mdp.n_actions)
        ]
    )


@pytest.mark.parametrize(
    "n_arms, n_states_list, truncate, exp_n_states, exp_n_actions",
    [
        (2, [2, 2], 3, 36, 2),
        (2, [2, 2], 5, 100, 2),
        (3, [2, 2, 2], 3, 216, 3),
        (3, [2, 3, 4], 5, 3_000, 3),
        (4, [2, 3, 4, 4], 4, 24_576, 4),
    ],
)
def test_mdp_convert_mdp_size(
    n_arms, n_states_list, truncate, exp_n_states, exp_n_actions
):
    print(n_states_list)
    restless_mab = RestlessMAB(
        [
            MarkovArm(n_states, np.eye(n_states), np.ones((n_states,)))
            for n_states in n_states_list
        ]
    )

    mdp = convert_to_mdp(restless_mab, truncate)

    assert mdp.n_states == exp_n_states
    assert mdp.n_actions == exp_n_actions


@pytest.mark.parametrize(
    "n_arms, p, q, truncate",
    [
        (2, 0.8, 0.6, 10),
        (2, 0.3, 0.8, 10),
        (3, 0.2, 0.9, 5),
    ],
)
def test_mdp_convert_equivalence(n_arms, p, q, truncate):
    # check that two equivalent MDPs (arising from the same ChannelAccessMAB) have the same optimal value function

    transition_matrix = np.array([[1 - q, q], [1 - p, p]])
    reward_vector = np.array([0, 1])

    channel_mab = ChannelAccessMAB(n_arms, p, q)
    channel_mdp = convert_to_mdp(channel_mab, truncate)
    channel_value = value_iteration(channel_mdp, 1e-3)

    restless_mab = SymmetricRestlessMAB(n_arms, transition_matrix, reward_vector)
    restless_mdp = convert_to_mdp(restless_mab, truncate)
    restless_value = value_iteration(restless_mdp, 1e-3)

    np.testing.assert_array_almost_equal(channel_value, restless_value, decimal=3)
