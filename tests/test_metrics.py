import numpy as np
import pytest

from hive.metrics import best_mass, consensus, l2_to_reference, t90


def test_best_mass_counts_selected_types():
    types = np.array([0, 1, 1, 1])
    assert best_mass(types, best=1) == pytest.approx(0.75)


def test_t90_finds_first_crossing_of_smoothed_curve():
    curve = np.concatenate([np.full(10, 0.1), np.full(10, 0.95)])
    step, hit = t90(curve, threshold=0.9, window=5)
    assert hit
    assert 10 <= step <= 15


def test_t90_reports_censoring_when_never_reached():
    step, hit = t90(np.full(50, 0.5), threshold=0.9)
    assert not hit
    assert step == 50


def test_consensus_is_max_policy_component():
    assert consensus(np.array([0.1, 0.7, 0.2])) == pytest.approx(0.7)


def test_l2_to_reference_is_rms_distance():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    assert l2_to_reference(a, b) == pytest.approx(np.sqrt(0.5))
