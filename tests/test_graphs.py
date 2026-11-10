import numpy as np
import pytest
from scipy import sparse

from hive.graphs import two_room_graph, degree_stats, is_connected, relaxation_time, rgg_graph, ws_graph


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


def test_large_cycle_uses_the_algebraic_spectral_branch():
    # The sparse branch used to ask for the largest-magnitude eigenvalue and
    # therefore returned lambda2 = -1 (tau = 0.5) for bipartite graphs.
    n = 600
    _, tau = relaxation_time(ws_graph(n, 2, 0.0, seed=0))
    assert 0.5 * n**2 / (2 * np.pi**2) < tau < 1.5 * n**2 / (2 * np.pi**2)


def test_two_node_graph_has_a_finite_relaxation_time():
    adj = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    lambda2, tau = relaxation_time(adj)
    assert lambda2 == pytest.approx(-1.0)
    assert tau == pytest.approx(0.5)


def test_dense_branch_uses_algebraic_lambda2_on_bipartite_graphs():
    # path 0-1-2-3 is bipartite, so its spectrum is symmetric about zero; the
    # second largest algebraic eigenvalue must come out positive, not -1.
    adj = sparse.csr_matrix(
        ([1.0] * 6, ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])), shape=(4, 4)
    )
    dense = np.asarray(adj.todense())
    root = np.sqrt(dense.sum(axis=1))
    expected = np.linalg.eigvalsh(dense / root[:, None] / root[None, :])[-2]
    lambda2, tau = relaxation_time(adj)
    assert expected > 0
    assert lambda2 == pytest.approx(expected, abs=1e-9)
    assert tau == pytest.approx(1.0 / (1.0 - expected), rel=1e-9)


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


def test_two_room_graph_has_bottleneck() -> None:
    adj = two_room_graph(50, bridge=2, seed=0)
    assert is_connected(adj)
    lambda2, tau = relaxation_time(adj)
    assert np.isfinite(lambda2)
    # Narrow bridge must inflate tau by an order of magnitude vs a clique.
    _, tau_clique = relaxation_time(complete_graph_adj(50))
    assert tau > 10 * tau_clique


def test_sparse_rooms_are_regular_except_at_bridge_endpoints() -> None:
    adj = two_room_graph(100, bridge=3, seed=1, room_degree=6)
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    assert is_connected(adj)
    assert set(np.unique(degrees)) <= {6, 7}
    assert (degrees == 7).sum() == 6  # three endpoints per room
    assert adj.diagonal().sum() == 0


def test_sparse_rooms_lower_tau_at_the_same_bridge_width() -> None:
    _, tau_clique = relaxation_time(two_room_graph(100, bridge=2, seed=0))
    _, tau_sparse = relaxation_time(two_room_graph(100, bridge=2, seed=0, room_degree=6))
    assert tau_sparse < tau_clique / 3


def test_room_degree_at_least_clique_size_falls_back_to_clique() -> None:
    clique = two_room_graph(40, bridge=2, seed=3)
    same = two_room_graph(40, bridge=2, seed=3, room_degree=19)
    assert (clique != same).nnz == 0


def test_sparse_rooms_reject_odd_degree_sum() -> None:
    with pytest.raises(ValueError):
        two_room_graph(42, bridge=2, seed=0, room_degree=5)  # 21 * 5 is odd
