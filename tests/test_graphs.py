import numpy as np
import pytest
from scipy import sparse

from hive.graphs import degree_stats, is_connected, relaxation_time, rgg_graph, ws_graph


def complete_graph_adj(n: int) -> sparse.csr_matrix:
    dense = np.ones((n, n)) - np.eye(n)
    return sparse.csr_matrix(dense)


def test_complete_graph_relaxes_in_about_one_step():
    lambda2, tau = relaxation_time(complete_graph_adj(50))
    assert tau == pytest.approx(1.0, abs=0.05)
    assert lambda2 == pytest.approx(-1.0 / 49.0, abs=0.01)


def test_cycle_relaxation_scales_quadratically():
    n = 200
    _, tau = relaxation_time(ws_graph(n, 2, 0.0, seed=0))
    assert 0.5 * n**2 / (2 * np.pi**2) < tau < 1.5 * n**2 / (2 * np.pi**2)


def test_ws_with_zero_rewiring_is_a_cycle():
    adj = ws_graph(20, 2, 0.0, seed=1)
    assert adj.nnz == 40
    assert np.allclose(np.asarray(adj.sum(axis=1)).ravel(), 2)


def test_disconnected_graph_returns_infinite_tau():
    adj = sparse.block_diag(
        [complete_graph_adj(5), complete_graph_adj(5)], format="csr"
    )
    assert not is_connected(adj)
    lambda2, tau = relaxation_time(adj)
    assert np.isinf(tau)
    assert lambda2 == 0.0


def test_large_radius_rgg_is_connected():
    adj = rgg_graph(80, radius=0.5, seed=2)
    assert is_connected(adj)
    stats = degree_stats(adj)
    assert stats["mean"] > 0
    assert stats["max"] >= stats["mean"] >= stats["min"]
