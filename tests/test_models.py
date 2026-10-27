import numpy as np
import pytest

from hive.bandit import make_q
from hive.models import mrd_step, p_cl_step, p_mcl_step, trd_step


def test_p_mcl_noise_free_matches_closed_form():
    # With deterministic rewards r_i = q_{a_i}, P-MCL reduces exactly to
    # pi'_a = pi_a q_a / sum_b pi_b q_b (the weighted-voter fixed point).
    q = make_q("middle")
    pi = np.full(10, 0.1)
    expected = pi * q / (pi @ q)
    new = p_mcl_step(pi, q, np.random.default_rng(1), B=20000, delta=0.0)
    assert np.allclose(new, expected, atol=0.01)


def test_p_cl_noise_free_matches_its_expectation():
    q = make_q("middle")
    pi = np.full(10, 0.1)
    expected = pi + pi * q - pi * (pi @ q)
    new = p_cl_step(pi, q, np.random.default_rng(1), B=20000, delta=0.0)
    assert np.allclose(new, expected, atol=0.01)


def test_trd_and_mrd_preserve_the_policy_simplex():
    q = make_q("low")
    pi = np.full(10, 0.1)
    for _ in range(20):
        pi = trd_step(pi, q, dt=0.1)
    assert pi.sum() == pytest.approx(1.0)
    for _ in range(20):
        pi = mrd_step(pi, q, dt=0.1)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= 0)


def test_mrd_concentrates_on_best_action():
    q = make_q("middle")
    pi = np.full(10, 0.1)
    for _ in range(100):
        pi = mrd_step(pi, q)
    assert pi.argmax() == 9
    assert pi[9] > 0.5


def test_mrd_survives_a_dead_type():
    q = make_q("middle")
    pi = np.zeros(10)
    pi[0] = 1.0
    out = mrd_step(pi, q)
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)
