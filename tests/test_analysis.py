import numpy as np
import pandas as pd
import pytest

from hive.analysis import hierarchical_bootstrap_law, law_diagnostics, law_frame
from hive.plots import fit_law


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "family": ["complete", "rgg", "rgg", "ws", "ws", "ws", "two_room"],
            "param": [1.0, 2.5, 2.0, 1.0, 0.2, 0.0, 1.0],
            "tau": [1.0, 34.0, 62.0, 3.6, 13.2, 2714.0, 31373.0],
            "path_length": [1.0, 4.16, 5.26, 3.69, 4.51, 42.08, 2.0],
            "degree_mean": [10.0, 10.5, 10.2, 6.0, 6.0, 6.0, 250.0],
            "degree_cv": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01],
            "connected": [True, True, True, True, True, True, True],
            "hit_rate": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "t90": [44.0, 48.0, 56.5, 58.5, 61.0, 173.5, 47.5],
        }
    )


def test_law_frame_excludes_intervention_and_censored_rows():
    summary = _summary()
    summary.loc[summary["family"] == "rgg", "hit_rate"] = 0.4
    summary.loc[summary["family"] == "ws", "connected"] = False
    frame = law_frame(summary)
    assert set(frame["family"]) == {"complete"}


def test_fit_law_ignores_disconnected_and_censored_rows():
    summary = _summary()
    summary.loc[summary["family"] == "two_room", "family"] = "ws"
    summary.loc[3, "connected"] = False
    summary.loc[4, "hit_rate"] = 0.2
    _, _, _, n_train = fit_law(summary)
    assert n_train == 5


def test_diagnostics_report_uncertainty_and_influence():
    diag = law_diagnostics(_summary(), n_boot=200)
    assert diag["n"] == 6
    assert np.isfinite(diag["intercept"])
    assert np.isfinite(diag["slope"])
    assert diag["ci"]["slope"][0] <= diag["ci"]["slope"][1]
    assert diag["prediction_at_L2"] > 0
    assert diag["influence"]
    assert diag["influence"][0]["label"].startswith(("ws", "rgg", "complete"))
    assert isinstance(diag["warnings"], list)
    assert diag["correlations"]["path_length"]["n"] == 6


def _seeds_frame(noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for L in (1.0, 2.0, 3.0, 4.0):
        for graph in range(2):
            offset = rng.normal(0.0, noise)
            for _ in range(3):
                rows.append(
                    {
                        "config": f"c{L:g}",
                        "graph_seed": graph,
                        "path_length": L,
                        "t90": 10.0 + 5.0 * L + offset + rng.normal(0.0, noise),
                    }
                )
    return pd.DataFrame(rows)


def test_hierarchical_bootstrap_recovers_an_exact_line():
    law = hierarchical_bootstrap_law(_seeds_frame(), n_boot=100)
    assert law["n_configs"] == 4
    assert law["n_graphs"] == 8
    assert law["n_seeds"] == 24
    assert law["slope"] == pytest.approx(5.0)
    assert law["intercept"] == pytest.approx(10.0)
    assert law["r2"] == pytest.approx(1.0)
    np.testing.assert_allclose(law["ci"]["slope"], [5.0, 5.0])


def test_hierarchical_bootstrap_ci_is_finite_and_ordered_under_noise():
    law = hierarchical_bootstrap_law(_seeds_frame(noise=3.0), n_boot=200)
    lo, hi = law["ci"]["slope"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi
    assert hi - lo > 0.0
    assert law["ci"]["r2"][0] <= law["ci"]["r2"][1]


def test_hierarchical_bootstrap_needs_at_least_three_configs():
    frame = _seeds_frame()
    law = hierarchical_bootstrap_law(frame[frame["config"].isin(["c1", "c2"])])
    assert law["n_configs"] == 2
    assert "error" in law
