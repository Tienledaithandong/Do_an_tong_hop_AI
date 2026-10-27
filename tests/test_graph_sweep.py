import numpy as np
import pytest
from scipy import sparse

from hive.bandit import make_q
from hive.graph_sweep import simulate_on_graph


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
