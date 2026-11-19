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


def test_reproduction_starts_each_learner_from_same_empirical_policy(monkeypatch):
    import hive.reproduce as reproduction

    pmcl_inputs = []
    mrd_inputs = []
    original_pmcl = reproduction.p_mcl_step
    original_mrd = reproduction.mrd_step

    def capture_pmcl(pi, *args, **kwargs):
        pmcl_inputs.append(pi.copy())
        return original_pmcl(pi, *args, **kwargs)

    def capture_mrd(pi, *args, **kwargs):
        mrd_inputs.append(pi.copy())
        return original_mrd(pi, *args, **kwargs)

    monkeypatch.setattr(reproduction, "p_mcl_step", capture_pmcl)
    monkeypatch.setattr(reproduction, "mrd_step", capture_mrd)
    out = run_reproduction(N=20, n=19, steps=1, seeds=30)

    np.testing.assert_array_equal(out["well_mixed"][:, 0], out["pmcl"][:, 0])
    np.testing.assert_array_equal(out["well_mixed"][:, 0], out["mrd"][:, 0])
    assert all(np.max(pi) == 0.10 for pi in pmcl_inputs)
    for pmcl, mrd in zip(pmcl_inputs, mrd_inputs, strict=True):
        np.testing.assert_array_equal(pmcl, mrd)
