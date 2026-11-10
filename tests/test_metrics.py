import numpy as np
import pytest

from hive.metrics import (
    THRESHOLD_METHODS,
    _smooth_causal,
    best_mass,
    consensus,
    kaplan_meier,
    l2_to_reference,
    rmst,
    t90,
    threshold_time,
)


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


def test_t90_does_not_censor_a_crossing_at_the_end_of_the_window():
    # Four consecutive hits at the end are enough for a centred 5-step
    # average to reach 0.9 once the boundary is padded with the edge value.
    curve = np.concatenate([np.full(396, 0.1), np.full(4, 1.0)])
    step, hit = t90(curve, threshold=0.9, window=5)
    assert hit
    assert step == 398


def test_consensus_is_max_policy_component():
    assert consensus(np.array([0.1, 0.7, 0.2])) == pytest.approx(0.7)


def test_l2_to_reference_is_rms_distance():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    assert l2_to_reference(a, b) == pytest.approx(np.sqrt(0.5))


def test_kaplan_meier_without_censoring_matches_empirical_survival():
    ts, ss = kaplan_meier(np.array([1.0, 2.0, 3.0]), np.ones(3, dtype=bool))
    np.testing.assert_allclose(ts, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(ss, [1.0, 2 / 3, 1 / 3, 0.0])


def test_kaplan_meier_groups_ties_and_censoring_only_shrinks_the_risk_set():
    # t=1: risk 4, one event -> 3/4; the run censored at 1 stays at risk.
    # t=2: risk 2, one event -> 3/8; t=3: risk 1, one event -> 0.
    times = np.array([1.0, 1.0, 2.0, 3.0])
    events = np.array([True, False, True, True])
    ts, ss = kaplan_meier(times, events)
    np.testing.assert_allclose(ts, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(ss, [1.0, 0.75, 0.375, 0.0])


def test_kaplan_meier_all_censored_stays_at_one():
    ts, ss = kaplan_meier(np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=bool))
    np.testing.assert_allclose(ts, [0.0])
    np.testing.assert_allclose(ss, [1.0])


def test_kaplan_meier_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        kaplan_meier(np.array([1.0, 2.0]), np.ones(3, dtype=bool))


def test_rmst_without_censoring_is_the_truncated_mean():
    times = np.array([1.0, 2.0, 3.0])
    events = np.ones(3, dtype=bool)
    assert rmst(times, events, horizon=5) == pytest.approx(2.0)
    assert rmst(times, events, horizon=2) == pytest.approx(5 / 3)


def test_rmst_all_censored_is_the_horizon():
    times = np.array([1.0, 2.0, 3.0])
    assert rmst(times, np.zeros(3, dtype=bool), horizon=5) == pytest.approx(5.0)


def test_rmst_mixed_censoring_uses_the_survival_area():
    # S = 1 on [0,1), 3/4 on [1,3), 3/8 on [3,4]: 1 + 1.5 + 0.375 = 2.875.
    times = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([True, False, True, False])
    assert rmst(times, events, horizon=4) == pytest.approx(2.875)


def test_rmst_empty_sample_is_nan():
    assert np.isnan(rmst(np.array([]), np.array([], dtype=bool), horizon=5))


def test_symmetric_smoother_crosses_before_the_raw_signal():
    # Reviewer's example: the centered 5-step average reaches 0.1 at step 1
    # even though the first nonzero value only appears at step 3.
    curve = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert threshold_time(curve, 0.1, 5, "symmetric") == (1.0, True)
    assert threshold_time(curve, 0.1, 5, "raw") == (3.0, True)
    assert threshold_time(curve, 0.1, 5, "causal") == (3.0, True)


def test_default_method_is_the_symmetric_smoother():
    curve = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert threshold_time(curve, 0.1, 5) == threshold_time(curve, 0.1, 5, "symmetric")
    assert t90(curve, 0.1, 5) == (1.0, True)


def test_causal_and_raw_localize_a_sustained_onset():
    # A step that jumps at index k and stays high: causal and raw report k,
    # while the centered smoother anticipates it.
    curve = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert threshold_time(curve, 0.1, 5, "raw") == (4.0, True)
    assert threshold_time(curve, 0.1, 5, "causal") == (4.0, True)
    assert threshold_time(curve, 0.1, 5, "symmetric") == (2.0, True)


def test_causal_smoothed_value_ignores_future_data():
    a = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    # Changing the tail must not alter the smoothed prefix up to the onset.
    np.testing.assert_allclose(_smooth_causal(a, 5)[:5], _smooth_causal(b, 5)[:5])
    assert threshold_time(a, 0.1, 5, "causal") == threshold_time(b, 0.1, 5, "causal")


def test_raw_method_is_the_first_index_at_threshold():
    curve = np.array([0.2, 0.5, 0.95, 0.95])
    assert threshold_time(curve, 0.9, 5, "raw") == (2.0, True)
    assert threshold_time(curve, 0.5, 5, "raw") == (1.0, True)


def test_every_method_reports_censoring_when_never_reached():
    curve = np.full(6, 0.5)
    for method in THRESHOLD_METHODS:
        assert threshold_time(curve, 0.9, 5, method) == (6.0, False)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        threshold_time(np.array([0.0, 1.0]), 0.1, 5, "bogus")


def test_t90_forwards_the_method_argument():
    curve = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert t90(curve, threshold=0.1, window=5, method="raw") == (3.0, True)
    assert t90(curve, threshold=0.1, window=5, method="causal") == (3.0, True)
