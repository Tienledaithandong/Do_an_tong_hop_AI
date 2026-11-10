import numpy as np
import pytest

from hive.bandit import Bandit, expected_rewards, make_q, sample_rewards


def test_q_middle_is_evenly_spaced_between_04_and_07():
    q = make_q("middle", 10)
    assert q[0] == pytest.approx(0.4)
    assert q[-1] == pytest.approx(0.7)
    assert np.allclose(np.diff(q), 0.3 / 9)


def test_rewards_stay_in_unit_interval_even_with_extreme_q():
    q = np.array([0.0, 1.0])
    actions = np.array([0, 1] * 5000)
    rewards = sample_rewards(q, actions, 0.3, np.random.default_rng(0))
    assert rewards.min() >= 0.0
    assert rewards.max() <= 1.0


def test_pull_centres_on_the_selected_actions_quality():
    q = make_q("middle")
    bandit = Bandit(q, delta=0.1, rng=np.random.default_rng(1))
    rewards = bandit.pull(np.full(5000, 5))
    assert rewards.mean() == pytest.approx(q[5], abs=0.01)


def test_best_is_the_argmax_action():
    assert Bandit(make_q("middle")).best == 9


def test_expected_rewards_is_identity_when_nothing_clips():
    q = make_q("middle", 10)
    assert np.allclose(expected_rewards(q, 0.1), q)


def test_expected_rewards_accounts_for_upper_clipping():
    # U(0.4, 1.4) clipped to [0, 1]: 60% mass on [0.4, 1.0] (mean 0.7),
    # 40% mass pinned at 1.0 -> 0.6 * 0.7 + 0.4 = 0.82.
    assert expected_rewards(np.array([0.9]), 0.5)[0] == pytest.approx(0.82)


def test_expected_rewards_accounts_for_lower_clipping():
    # U(-0.4, 0.6) clipped to [0, 1]: 40% mass pinned at 0.0,
    # 60% mass on [0.0, 0.6] (mean 0.3) -> 0.6 * 0.3 = 0.18.
    assert expected_rewards(np.array([0.1]), 0.5)[0] == pytest.approx(0.18)


def test_expected_rewards_matches_monte_carlo_of_sampled_rewards():
    q = np.array([0.1, 0.5, 0.9])
    actions = np.arange(3).repeat(40000)
    rewards = sample_rewards(q, actions, 0.5, np.random.default_rng(7))
    means = np.array([rewards[actions == a].mean() for a in range(3)])
    assert np.allclose(means, expected_rewards(q, 0.5), atol=0.01)
