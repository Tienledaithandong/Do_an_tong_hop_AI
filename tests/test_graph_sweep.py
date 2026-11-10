import numpy as np
import pytest
from scipy import sparse

from hive.bandit import make_q
from hive.graph_sweep import reference_mrd, simulate_on_graph
from hive.graphs import average_path_length, relaxation_time, two_room_graph


def complete_graph_adj(n: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(np.ones((n, n)) - np.eye(n))


def test_complete_graph_swarm_learns_the_best_option():
    adj = complete_graph_adj(60)
    res = simulate_on_graph(adj, make_q("middle"), steps=80, seeds=3, n_types=10)
    assert res["t90"].mean() < 60
    assert res["final"].mean() > 0.7


def test_disconnected_graph_runs_without_nan():
    adj = sparse.block_diag(
        [complete_graph_adj(15), complete_graph_adj(15)], format="csr"
    )
    res = simulate_on_graph(adj, make_q("middle"), steps=40, seeds=2, n_types=10)
    assert np.isfinite(res["curves"]).all()
    assert res["curves"].shape == (2, 41)


def test_two_room_bottleneck_does_not_slow_learning():
    # Huge relaxation time but a short path length: weighted imitation
    # follows distance, not mixing, so t90 must stay low.
    adj = two_room_graph(120, bridge=2, seed=0)
    _, tau = relaxation_time(adj)
    path = average_path_length(adj)
    assert np.isfinite(tau) and tau > 100.0
    assert path < 3.0
    res = simulate_on_graph(adj, make_q("middle"), steps=120, seeds=4, n_types=10)
    assert np.isfinite(res["curves"]).all()
    assert res["t90"].mean() < 90.0


def test_mrd_reference_shares_each_seeds_initial_distribution():
    # N not divisible by n: the empirical start is not uniform and differs
    # per seed, so the averaged MRD must start at the averaged empirical mass,
    # not at 1/n.
    adj = complete_graph_adj(25)
    res = simulate_on_graph(adj, make_q("middle"), steps=20, seeds=5, n_types=10)
    assert res["mrd"][0] == pytest.approx(res["curves"][:, 0].mean(), abs=1e-12)
    assert res["mrd"][0] != pytest.approx(0.1, abs=1e-9)


def test_mrd_reference_uses_the_clipped_reward_expectation():
    # high scenario with delta=0.5 pushes q+delta past 1, so the reference must
    # use E[clip(...)] rather than raw q. With N divisible by n every seed starts
    # uniform, so the averaged MRD equals the uniform-init clipped reference.
    q = make_q("high")
    adj = complete_graph_adj(30)
    res = simulate_on_graph(adj, q, steps=40, seeds=3, delta=0.5, n_types=10)
    clipped = reference_mrd(q, 40, delta=0.5)
    raw = reference_mrd(q, 40, delta=0.0)
    assert np.allclose(res["mrd"], clipped, atol=1e-12)
    assert not np.allclose(clipped, raw, atol=1e-3)


def test_seed_offset_shifts_the_rng_without_changing_determinism():
    adj = complete_graph_adj(30)
    q = make_q("middle")
    a = simulate_on_graph(adj, q, steps=20, seeds=3, n_types=10, seed_offset=0)
    b = simulate_on_graph(adj, q, steps=20, seeds=3, n_types=10, seed_offset=0)
    c = simulate_on_graph(adj, q, steps=20, seeds=3, n_types=10, seed_offset=100)
    np.testing.assert_allclose(a["curves"], b["curves"])
    assert not np.allclose(a["curves"], c["curves"])
    # offset=1, seed 0 is the same RNG stream as offset=0, seed 1.
    d = simulate_on_graph(adj, q, steps=20, seeds=1, n_types=10, seed_offset=1)
    np.testing.assert_allclose(d["curves"][0], a["curves"][1])
