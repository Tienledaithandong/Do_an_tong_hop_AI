import numpy as np
import pytest
from scipy import sparse

from hive.bandit import make_q
from hive.graph_sweep import simulate_on_graph
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
