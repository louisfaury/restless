
import numpy as np
import pytest

from restless.envs import ChannelAccessMAB
from restless.envs.mdp_convert import convert_channel_to_mdp


@pytest.mark.parametrize("n_arms, p, q, truncate, exp_n_states, exp_n_actions",
                         [
                             (2, 0.8, 0.4, 1, 4, 2),
                             (2, 0.8, 0.4, 2, 16, 2),
                             (2, 0.8, 0.4, 10, 20**2, 2),
                             (3, 0.8, 0.4, 10, 20**3, 3),
                            (2, 0.4, 0.8, 2, 16, 2),
                         ]
                         )
def test_channel_converter(n_arms, p, q, truncate, exp_n_states, exp_n_actions):
    channel_mab = ChannelAccessMAB(n_arms=n_arms, p=p, q=q)

    mdp = convert_channel_to_mdp(channel_mab, truncate)

    assert mdp.n_states == exp_n_states
    assert mdp.n_actions == exp_n_actions

    assert np.all([mdp.reward_function[k].min() == min([p,q]) for k in range(mdp.n_actions)])
    assert np.all([mdp.reward_function[k].max() == max([p, q]) for k in range(mdp.n_actions)])

    assert np.all([mdp.transition_kernel[k].max() == max([p, 1-p, q, 1-q]) for k in range(mdp.n_actions)])