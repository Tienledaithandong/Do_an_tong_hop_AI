import numpy as np
import pytest

from hive.bridge import (
    arrival_events,
    bootstrap_exponent,
    bootstrap_ci,
    censored_median,
    first_arrival,
    fit_loglog,
    fit_loglog2,
    run_replicates,
    summarise,
)


def test_first_arrival_marks_unreached_seeds_as_censored():
    curves = np.array(
        [
            [0.0, 0.0, 0.1, 0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    step, hit = first_arrival(curves)
    assert step.tolist() == [2.0, 4.0]
    assert hit.tolist() == [True, False]


def test_arrival_events_counts_invasions_and_extinctions():
    curves = np.array(
        [
            [0.0, 0.1, 0.0, 0.2, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    arrivals, extinctions = arrival_events(curves)
    assert arrivals.tolist() == [3, 0]
    assert extinctions.tolist() == [2, 0]


def test_censored_median_is_infinite_when_most_seeds_miss():
    values = np.array([10.0, 20.0, 99.0, 99.0])
    hit = np.array([True, True, False, False])
    assert censored_median(values, hit) == np.inf
    assert censored_median(values[:3], np.array([True, True, True])) == 20.0


def test_censored_median_ignores_value_of_unreached_seeds():
    values = np.array([5.0, 7.0, 1.0])
    hit = np.array([True, True, False])
    assert censored_median(values, hit) == 7.0


def test_bootstrap_ci_brackets_the_median_and_handles_clusters():
    values = np.arange(1.0, 41.0)
    hit = np.ones(40, dtype=bool)
    cluster = np.repeat(np.arange(4), 10)
    lo, hi = bootstrap_ci(values, hit, n_boot=200, cluster=cluster)
    assert lo <= censored_median(values, hit) <= hi
    lo2, hi2 = bootstrap_ci(values, hit, n_boot=200)
    assert (lo2, hi2) != (lo, hi) or lo == hi


def test_fit_loglog_recovers_a_power_law():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    fit = fit_loglog(x, 3.0 * x**-1.0)
    assert fit["slope"] == pytest.approx(-1.0)
    assert fit["r2"] == pytest.approx(1.0)


def test_fit_loglog2_recovers_conductance_scaling():
    bridge = np.array([1.0, 2.0, 4.0, 1.0, 2.0, 4.0])
    size = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
    y = 5.0 * (2 * bridge / size) ** -0.7
    fit = fit_loglog2(bridge, size, y)
    assert fit["bridge_exp"] == pytest.approx(-0.7)
    assert fit["size_exp"] == pytest.approx(0.7)
    assert fit["r2"] == pytest.approx(1.0)


def test_bootstrap_exponent_brackets_the_slope_on_a_clean_power_law():
    import pandas as pd

    rows = []
    for N in (100, 200):
        for bridge in (1, 2, 4, 8):
            phi = bridge / (N / 2)
            for graph_seed in range(3):
                for seed in range(5):
                    rows.append(
                        {
                            "N": N,
                            "bridge": bridge,
                            "phi": phi,
                            "graph_seed": graph_seed,
                            "seed": seed,
                            "t": 4.0 * phi**-0.7,
                            "hit": True,
                        }
                    )
    lo, hi = bootstrap_exponent(pd.DataFrame(rows), "t", "hit", n_boot=40)
    assert lo == pytest.approx(-0.7) and hi == pytest.approx(-0.7)


def test_run_replicates_shapes_and_summary():
    df = run_replicates({40: 60}, [2, 6], graph_seeds=2, seeds=3, n_types=5)
    assert len(df) == 1 * 2 * 2 * 3
    assert set(df["bridge"]) == {2, 6}
    assert (df["phi"] == df["bridge"] / 20).all()
    assert df["t_first"].min() >= 1
    summary = summarise(df, n_boot=50)
    assert list(summary["bridge"]) == [2, 6]
    assert (summary["graphs"] == 2).all()
    assert (summary["runs"] == 6).all()
    assert (df["arrivals"] >= df["extinctions"]).all()
    assert {"arrivals_mean", "extinctions_mean"} <= set(summary.columns)


def test_replicate_figure_is_written(tmp_path):
    from hive.plots import plot_two_room_replicates

    df = run_replicates({40: 60, 60: 80}, [2, 6], graph_seeds=2, seeds=3, n_types=5)
    paths = plot_two_room_replicates(summarise(df, n_boot=30), tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
