import numpy as np
import pytest
from scipy import sparse

from hive.bandit import make_q
from hive.graph_sweep import reference_mrd, simulate_on_graph
from hive.graphs import ws_graph
from hive.reproduce import run_reproduction


def _ring(n: int = 40) -> sparse.csr_matrix:
    return ws_graph(n, 4, 0.1, seed=0)


def test_graph_simulation_is_deterministic_for_a_fixed_seed():
    """The same graph and seed sequence must reproduce identical curves."""
    adj = _ring()
    q = make_q("middle", 5)
    first = simulate_on_graph(adj, q, steps=15, seeds=3, n_types=5)
    second = simulate_on_graph(adj, q, steps=15, seeds=3, n_types=5)
    assert np.array_equal(first["curves"], second["curves"])
    assert np.array_equal(first["t90"], second["t90"])


def test_progress_callback_reports_every_seed():
    adj = _ring()
    calls: list[tuple[int, int]] = []
    simulate_on_graph(
        adj,
        make_q("middle", 5),
        steps=5,
        seeds=3,
        n_types=5,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_return_state_exposes_the_last_seed_types():
    adj = _ring()
    without = simulate_on_graph(adj, make_q("middle", 5), steps=3, seeds=2, n_types=5)
    with_state = simulate_on_graph(
        adj, make_q("middle", 5), steps=3, seeds=2, n_types=5, return_state=True
    )
    assert "final_types" not in without
    types = with_state["final_types"]
    assert types.shape == (adj.shape[0],)
    assert types.min() >= 0 and types.max() < 5


def test_curves_stay_inside_the_unit_interval():
    result = simulate_on_graph(_ring(), make_q("middle", 5), steps=20, seeds=2, n_types=5)
    assert result["curves"].min() >= 0.0
    assert result["curves"].max() <= 1.0
    assert result["curves"].shape == (2, 21)
    assert np.isfinite(result["curves"]).all()


def test_reference_mrd_is_monotone_for_the_best_action():
    curve = reference_mrd(make_q("middle", 5), steps=100)
    assert curve[0] == pytest.approx(0.2)
    assert np.all(np.diff(curve) >= -1e-12)
    assert curve[-1] > 0.9


def test_reproduction_progress_callback_reports_every_seed():
    calls: list[tuple[int, int]] = []
    out = run_reproduction(
        N=40,
        steps=5,
        seeds=3,
        n=5,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert out["well_mixed"].shape == (3, 6)


def test_complete_graph_voter_matches_the_well_mixed_step():
    """On a graph without self-loops the complete topology is well-mixed."""
    n = 30
    n_types = 5
    adj = sparse.csr_matrix(np.ones((n, n)) - np.eye(n))
    rng = np.random.default_rng(7)
    types = rng.integers(0, n_types, size=n)
    rewards = rng.random(n) + 0.05

    from hive.swarm import graph_voter, well_mixed_voter

    graph = graph_voter(types, rewards, adj, np.random.default_rng(3), n_types=n_types)
    mixed = well_mixed_voter(types, rewards, np.random.default_rng(3), n_types=n_types)
    assert np.array_equal(graph, mixed)
