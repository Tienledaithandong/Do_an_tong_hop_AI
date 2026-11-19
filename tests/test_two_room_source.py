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

    def spy(q, steps, pi0=None, delta=0.0):
        seen.append(pi0)
        return real(q, steps, pi0=pi0, delta=delta)

    monkeypatch.setattr(gs, "reference_mrd", spy)
    N, n = 40, 5
    q = make_q("middle", n)
    result = gs.simulate_two_room_source(
        two_room_graph(N, bridge=3, seed=0), q, steps=10, seeds=2, n_types=n
    )
    # Both seeds have identical initial counts, so the deterministic reference
    # is computed once and reused, even though the node assignments differ.
    assert len(seen) == 1
    pi0 = seen[0]
    assert pi0 is not None
    assert pi0.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi0[int(np.argmax(q))], result["curves"][:, 0])


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


def test_source_room_seed_offset_selects_same_per_seed_stream():
    adj = two_room_graph(40, bridge=3, seed=0)
    q = make_q("middle", 5)
    multiple = simulate_two_room_source(adj, q, steps=50, seeds=3, n_types=5)
    shifted = simulate_two_room_source(adj, q, steps=50, seeds=1, n_types=5, seed_offset=1)
    for key in ("curves", "source_curves", "target_curves"):
        np.testing.assert_array_equal(shifted[key][0], multiple[key][1])


def test_absorbing_consensus_shortcut_matches_every_unoptimized_step(monkeypatch):
    import hive.graph_sweep as gs
    from hive.bandit import Bandit
    from hive.swarm import graph_voter

    N, n, steps = 12, 2, 160
    adj = two_room_graph(N, bridge=3, seed=0)
    q = make_q("middle", n)
    calls = 0

    def count_updates(*args, **kwargs):
        nonlocal calls
        calls += 1
        return graph_voter(*args, **kwargs)

    monkeypatch.setattr(gs, "graph_voter", count_updates)
    actual = gs.simulate_two_room_source(
        adj, q, steps=steps, seeds=1, n_types=n, return_state=True
    )
    assert calls < steps

    rng = np.random.default_rng(0)
    bandit = Bandit(q, delta=0.1, rng=rng)
    mask = room_mask(N)
    types = initial_types_source_room(rng, n, N, int(np.argmax(q)), mask)
    expected = [[], [], []]
    for t in range(steps + 1):
        best = types == np.argmax(q)
        expected[0].append(best.mean())
        expected[1].append(best[mask].mean())
        expected[2].append(best[~mask].mean())
        if t < steps:
            types = graph_voter(types, bandit.pull(types), adj, rng, n_types=n)
    for key, values in zip(("curves", "source_curves", "target_curves"), expected):
        np.testing.assert_array_equal(actual[key][0], values)
    np.testing.assert_array_equal(actual["final_types"], types)
