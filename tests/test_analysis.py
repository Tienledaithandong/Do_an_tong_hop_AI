import numpy as np
import pandas as pd

from hive.analysis import law_diagnostics, law_frame
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
