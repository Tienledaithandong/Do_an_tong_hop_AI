import numpy as np
import pytest

from hive.bandit import Bandit, make_q, sample_rewards


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
