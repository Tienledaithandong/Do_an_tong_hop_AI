import numpy as np
import pytest

from hive.reproduce import run_reproduction


def test_reproduction_curves_have_expected_shape():
    out = run_reproduction(N=100, steps=20, seeds=2)
    for key in ("well_mixed", "pmcl", "mrd"):
        assert out[key].shape == (2, 21)


def test_pmcl_tracks_mrd_and_learning_happens():
    out = run_reproduction(N=200, steps=60, seeds=2)
    final_mrd = out["mrd"][:, -1].mean()
    final_pmcl = out["pmcl"][:, -1].mean()
    final_wv = out["well_mixed"][:, -1].mean()
    assert final_mrd > 0.7
    assert abs(final_pmcl - final_mrd) < 0.2
    assert final_wv > 0.6
