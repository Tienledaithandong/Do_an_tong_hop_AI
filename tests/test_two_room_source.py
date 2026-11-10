import numpy as np
import pytest

from hive.bandit import make_q
from hive.graph_sweep import (
    initial_types_source_room,
    reference_mrd,
    room_mask,
    simulate_two_room_source,
)
from hive.graphs import two_room_graph


def test_reference_mrd_honours_a_non_uniform_initial_policy():
    q = make_q("middle", 5)
    best = int(np.argmax(q))
    pi0 = np.zeros(5)
    pi0[best] = 0.05
    pi0[(best + 1) % 5] = 0.95
    curve = reference_mrd(q, steps=20, pi0=pi0)
    assert curve[0] == pytest.approx(0.05)
    assert curve[-1] > curve[0]


def test_source_room_mrd_baseline_shares_the_initial_condition(monkeypatch):
    import hive.graph_sweep as gs

    seen = []
    real = gs.reference_mrd

    def spy(q, steps, pi0=None):
        seen.append(pi0)
        return real(q, steps, pi0=pi0)

    monkeypatch.setattr(gs, "reference_mrd", spy)
    N, n = 40, 5
    q = make_q("middle", n)
    result = gs.simulate_two_room_source(
        two_room_graph(N, bridge=3, seed=0), q, steps=10, seeds=2, n_types=n
    )
    assert len(seen) == 2
    for seed, pi0 in enumerate(seen):
        assert pi0 is not None
        assert pi0.sum() == pytest.approx(1.0)
        assert pi0[int(np.argmax(q))] == pytest.approx(result["curves"][seed, 0])


def test_source_initializer_keeps_best_out_of_the_target_room():
    N, n = 40, 5
    q = make_q("middle", n)
    best = int(np.argmax(q))
    source = room_mask(N, 0)
    types = initial_types_source_room(np.random.default_rng(0), n, N, best, source)
    assert np.any(types[source] == best)
    assert not np.any(types[~source] == best)
    assert types.size == N
    assert set(np.unique(types)) <= set(range(n))


def test_target_room_starts_without_the_best_option():
    N, n = 40, 5
    q = make_q("middle", n)
    result = simulate_two_room_source(
        two_room_graph(N, bridge=3, seed=0),
        q,
        steps=10,
        seeds=2,
        n_types=n,
    )
    assert result["target_curves"][:, 0].tolist() == [0.0, 0.0]
    assert np.all(result["source_curves"][:, 0] > 0)
    assert result["target_curves"].shape == (2, 11)
    assert result["t50_target"].shape == (2,)


def test_source_room_experiment_reports_thresholds_and_hits():
    N, n = 60, 5
    q = make_q("middle", n)
    result = simulate_two_room_source(
        two_room_graph(N, bridge=6, seed=0),
        q,
        steps=200,
        seeds=3,
        n_types=n,
        source_room=1,
    )
    assert np.all(result["final_target"] >= 0)
    assert np.all(result["final_target"] <= 1)
    assert result["hit50_target"].dtype == bool
    assert result["t50_target"].shape == (3,)
    assert result["curves"].shape == (3, 201)
