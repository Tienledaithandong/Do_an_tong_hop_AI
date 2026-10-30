import numpy as np
import pytest
from scipy import sparse

from hive.graphs import average_path_length


def test_complete_graph_has_unit_average_path_length():
    n = 20
    adj = sparse.csr_matrix(np.ones((n, n)) - np.eye(n))
    assert average_path_length(adj) == pytest.approx(1.0)


def test_path_graph_has_the_closed_form_mean_distance():
    # Path on 7 nodes: mean pairwise distance = (n + 1) / 3
    n = 7
    rows = np.arange(n - 1)
    adj = sparse.csr_matrix(
        (np.ones(n - 1), (rows, rows + 1)), shape=(n, n)
    )
    adj = (adj + adj.T).tocsr()
    assert average_path_length(adj) == pytest.approx((n + 1) / 3)


def test_disconnected_graph_averages_only_over_reachable_pairs():
    adj = sparse.block_diag(
        [sparse.csr_matrix(np.ones((2, 2)) - np.eye(2))] * 2, format="csr"
    )
    assert average_path_length(adj) == pytest.approx(1.0)


def test_fragmented_graph_with_unequal_components_ignores_unreachable_pairs():
    # A 3-clique plus two isolated nodes. Only the 6 ordered pairs inside the
    # clique are reachable, each at distance 1, so the mean is 1.0.
    dense = np.zeros((5, 5))
    dense[:3, :3] = np.ones((3, 3)) - np.eye(3)
    adj = sparse.csr_matrix(dense)
    assert average_path_length(adj) == pytest.approx(1.0)


def test_large_component_plus_isolate_is_not_inflated():
    # 4-clique plus one isolate: reachable off-diagonal pairs = 12 at distance 1.
    dense = np.zeros((5, 5))
    dense[:4, :4] = np.ones((4, 4)) - np.eye(4)
    adj = sparse.csr_matrix(dense)
    assert average_path_length(adj) == pytest.approx(1.0)
