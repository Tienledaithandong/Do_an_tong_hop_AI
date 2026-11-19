import numpy as np
import pandas as pd
import pytest

from hive.plots import fit_law, fit_law_sensitivity


def test_fit_law_excludes_the_two_room_intervention():
    summary = pd.DataFrame(
        {
            "family": ["ws", "rgg", "ws", "two_room"],
            "path_length": [2.0, 4.0, 6.0, 2.0],
            "t90": [50.0, 60.0, 70.0, 47.5],
        }
    )
    intercept, slope, r2, n_train = fit_law(summary)
    assert n_train == 3
    assert intercept == pytest.approx(40.0)
    assert slope == pytest.approx(5.0)
    assert r2 == pytest.approx(1.0)


def test_fit_law_sensitivity_reports_inclusion_counts_and_refits():
    summary = pd.DataFrame(
        {
            "family": ["ws"] * 5,
            "path_length": [1.0, 2.0, 3.0, 4.0, 5.0],
            "t90": [15.0, 20.0, 25.0, 120.0, 150.0],
            "hit_rate": [1.0, 0.8, 0.6, 0.4, 0.0],
        }
    )
    rows = fit_law_sensitivity(summary, thresholds=(0.0, 0.5, 0.75))
    assert [row["n_train"] for row in rows] == [5, 3, 2]
    assert rows[0]["slope"] != pytest.approx(rows[1]["slope"])
    assert np.isnan(rows[2]["slope"])


def test_fit_law_rejects_hit_rate_cutoffs_outside_unit_interval():
    with pytest.raises(ValueError, match="min_hit_rate"):
        fit_law(pd.DataFrame({"family": ["ws"], "path_length": [1], "t90": [1]}), 1.1)
