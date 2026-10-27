"""Communication graphs and their relaxation time.

``tau = 1 / (1 - lambda2)`` is the relaxation time of the random walk on the
graph, where ``lambda2`` is the second largest eigenvalue of the symmetric
normalised adjacency ``S = D^-1/2 A D^-1/2``. ``tau`` is the natural clock of
consensus-like dynamics: ``tau ~ 1`` means a well-mixed population, large
``tau`` means information crawls.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def ws_graph(n: int, k: int, p: float, seed: int | None = None) -> sparse.csr_matrix:
    """Watts-Strogatz small-world graph; ``p=0`` is a ring lattice."""
    graph = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return nx.to_scipy_sparse_array(
        graph, nodelist=list(range(n)), dtype=float, format="csr"
    )


def rgg_graph(n: int, radius: float, seed: int | None = None) -> sparse.csr_matrix:
    """Random geometric graph on the unit square."""
    graph = nx.random_geometric_graph(n, radius, seed=seed)
    return nx.to_scipy_sparse_array(
        graph, nodelist=list(range(n)), dtype=float, format="csr"
    )


def is_connected(adj: sparse.spmatrix) -> bool:
    adj = sparse.csr_matrix(adj)
    if adj.shape[0] == 0:
        return False
    n_components, _ = sparse.csgraph.connected_components(adj, directed=False)
    return bool(n_components == 1)


def degree_stats(adj: sparse.spmatrix) -> dict[str, float]:
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    mean = float(degrees.mean()) if degrees.size else 0.0
    std = float(degrees.std()) if degrees.size else 0.0
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean > 0 else float("inf"),
        "min": float(degrees.min()) if degrees.size else 0.0,
        "max": float(degrees.max()) if degrees.size else 0.0,
    }


def average_path_length(adj: sparse.spmatrix) -> float:
    """Mean shortest-path length over all reachable ordered pairs.

    Disconnected graphs average only over pairs that can reach each other, so a
    fragmented swarm still yields a finite (but smaller) distance scale.
    """
    adj = sparse.csr_matrix(adj)
    if adj.shape[0] < 2:
        return 0.0
    distances = sparse.csgraph.shortest_path(adj, unweighted=True, directed=False)
    finite = distances[np.isfinite(distances)]
    off_diagonal = finite.size - np.isfinite(np.diag(distances)).sum()
    if off_diagonal <= 0:
        return 0.0
    return float((finite.sum() - 0.0) / off_diagonal)


def two_room_graph(n: int, bridge: int = 3, seed: int | None = None) -> sparse.csr_matrix:
    """Two dense rooms joined by a narrow bridge (a mixing bottleneck).

    The narrow bridge dramatically slows random-walk mixing (large tau) while
    only modestly increasing shortest-path length (L), so tau and L decouple.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    n1, n2 = half, n - half

    k = min(bridge, n1, n2)
    bridge_a = rng.choice(n1, size=k, replace=False)
    bridge_b = rng.choice(n2, size=k, replace=False)

    row: list[int] = []
    col: list[int] = []

    def add(i: int, j: int) -> None:
        row.append(i)
        col.append(j)
        row.append(j)
        col.append(i)

    # Room A: complete graph on [0, n1)
    for i in range(n1):
        for j in range(i + 1, n1):
            add(i, j)
    # Room B: complete graph on [n1, n)
    for i in range(n1, n):
        for j in range(i + 1, n):
            add(i, j)
    # Bridge: connect corresponding endpoints between rooms
    for a, b in zip(bridge_a, bridge_b):
        add(a, n1 + b)

    return sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))


def relaxation_time(adj: sparse.spmatrix) -> tuple[float, float]:
    """Return ``(lambda2, tau)``; disconnected graphs give ``tau = inf``."""
    adj = sparse.csr_matrix(adj, dtype=float)
    n = adj.shape[0]
    if n < 3 or not is_connected(adj):
        return 0.0, float("inf")

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    if np.any(degrees <= 0):
        return 0.0, float("inf")

    inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    normalised = (inv_sqrt @ adj @ inv_sqrt).tocsr()

    if n <= 512:
        eigenvalues = np.linalg.eigvalsh(normalised.toarray())
        lambda2 = float(eigenvalues[-2])
    else:
        eigenvalues = spla.eigsh(normalised, k=2, which="LM", return_eigenvectors=False)
        lambda2 = float(np.sort(eigenvalues)[-2])

    lambda2 = min(lambda2, 1.0 - 1e-12)
    return lambda2, float(1.0 / (1.0 - lambda2))
