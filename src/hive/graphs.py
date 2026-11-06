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

__all__ = [
    "ws_graph",
    "rgg_graph",
    "is_connected",
    "degree_stats",
    "average_path_length",
    "two_room_graph",
    "two_room_hetero",
    "relaxation_time",
]


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

    Unreachable pairs and the zero diagonal are excluded, so a fragmented
    swarm yields a finite distance scale over reachable pairs only.
    """
    adj = sparse.csr_matrix(adj)
    if adj.shape[0] < 2:
        return 0.0
    distances = sparse.csgraph.shortest_path(adj, unweighted=True, directed=False)
    # Average only over reachable off-diagonal pairs; mask the zero diagonal
    # and the inf entries of unreachable pairs (exact for fragmented graphs).
    dist = np.asarray(distances, dtype=float).copy()
    np.fill_diagonal(dist, np.nan)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def two_room_graph(
    n: int,
    bridge: int = 3,
    seed: int | None = None,
    room_degree: int | None = None,
) -> sparse.csr_matrix:
    """Two rooms joined by a narrow bridge (a mixing bottleneck).

    The narrow bridge dramatically slows random-walk mixing (large tau) while
    only modestly increasing shortest-path length (L), so tau and L decouple.

    By default each room is a clique. With ``room_degree=d`` each room is a
    random ``d``-regular graph instead. At a fixed bridge width this keeps the
    bridge-per-node ratio unchanged while lowering ``tau`` roughly in
    proportion to ``d / room_size``, which lets experiments tell the two apart.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    n1, n2 = half, n - half

    k = min(bridge, n1, n2)
    bridge_a = rng.choice(n1, size=k, replace=False)
    bridge_b = rng.choice(n2, size=k, replace=False)

    if room_degree is not None and room_degree < min(n1, n2) - 1:
        return _two_sparse_rooms(n1, n2, room_degree, bridge_a, bridge_b, rng)

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


def _two_sparse_rooms(
    n1: int,
    n2: int,
    degree: int,
    bridge_a: np.ndarray,
    bridge_b: np.ndarray,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    if degree < 2:
        raise ValueError("room_degree must be at least 2")
    if (n1 * degree) % 2 or (n2 * degree) % 2:
        raise ValueError("room size times room_degree must be even")
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for offset, size in ((0, n1), (n1, n2)):
        room = nx.random_regular_graph(degree, size, seed=int(rng.integers(2**31 - 1)))
        edges = np.array(room.edges(), dtype=int) + offset
        rows.extend([edges[:, 0], edges[:, 1]])
        cols.extend([edges[:, 1], edges[:, 0]])
    rows.extend([bridge_a, n1 + bridge_b])
    cols.extend([n1 + bridge_b, bridge_a])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    return sparse.csr_matrix((np.ones(row.size), (row, col)), shape=(n1 + n2, n1 + n2))


PLACEMENTS = ("random", "hub", "leaf")


def _pick_endpoints(
    degrees: np.ndarray, k: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Choose ``k`` distinct bridge endpoints in one room by degree rank.

    ``hub`` takes the highest-degree nodes, ``leaf`` the lowest, and
    ``random`` a uniform sample. Ties are broken at random.
    """
    if mode not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {mode!r}")
    if mode == "random":
        return rng.choice(degrees.size, size=k, replace=False)
    tie = rng.random(degrees.size)
    order = np.lexsort((tie, -degrees if mode == "hub" else degrees))
    return order[:k]


def two_room_hetero(
    n: int,
    bridge: int = 3,
    seed: int | None = None,
    kind: str = "ba",
    m: int = 3,
    source: str = "random",
    target: str = "random",
) -> sparse.csr_matrix:
    """Two rooms with uneven degrees and controllable bridge endpoints.

    ``kind="ba"`` builds each room as a Barabasi-Albert graph (mean degree
    about ``2m``, a few hubs); ``kind="regular"`` builds a ``2m``-regular
    graph with the same mean degree as a control. ``source`` and ``target``
    choose whether the bridge touches hubs, leaves or random nodes of each
    room. Room 0 is the source (nodes ``[0, n // 2)``).
    """
    if kind not in ("ba", "regular"):
        raise ValueError("kind must be 'ba' or 'regular'")
    rng = np.random.default_rng(seed)
    half = n // 2
    sizes = (half, n - half)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    room_degrees: list[np.ndarray] = []
    for offset, size in zip((0, half), sizes):
        sub_seed = int(rng.integers(2**31 - 1))
        if kind == "ba":
            room = nx.barabasi_albert_graph(size, m, seed=sub_seed)
        else:
            room = nx.random_regular_graph(2 * m, size, seed=sub_seed)
        edges = np.array(room.edges(), dtype=int) + offset
        rows.extend([edges[:, 0], edges[:, 1]])
        cols.extend([edges[:, 1], edges[:, 0]])
        room_degrees.append(np.array([room.degree(i) for i in range(size)]))

    k = min(bridge, *sizes)
    src = _pick_endpoints(room_degrees[0], k, source, rng)
    tgt = _pick_endpoints(room_degrees[1], k, target, rng) + half
    rows.extend([src, tgt])
    cols.extend([tgt, src])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    return sparse.csr_matrix((np.ones(row.size), (row, col)), shape=(n, n))


def relaxation_time(adj: sparse.spmatrix) -> tuple[float, float]:
    """Return ``(lambda2, tau)``; disconnected graphs give ``tau = inf``."""
    adj = sparse.csr_matrix(adj, dtype=float)
    n = adj.shape[0]
    if n < 2 or not is_connected(adj):
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
        # ``which="LM"`` returns the largest *magnitude* eigenvalues; on a
        # bipartite graph that is 1 and -1, so lambda2 comes out as -1. Ask
        # for the largest algebraic eigenvalues instead.
        eigenvalues = spla.eigsh(normalised, k=2, which="LA", return_eigenvectors=False)
        lambda2 = float(np.sort(eigenvalues)[-2])

    lambda2 = min(lambda2, 1.0 - 1e-12)
    return lambda2, float(1.0 / (1.0 - lambda2))
