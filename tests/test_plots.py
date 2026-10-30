import pandas as pd
import pytest

from hive.plots import fit_law


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
