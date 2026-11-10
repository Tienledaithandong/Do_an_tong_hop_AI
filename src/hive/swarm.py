"""Weighted voter swarm models.

Every individual casts a vote for its current type weighted by the reward it
just observed. An individual then copies the type of a vote drawn
proportionally to weight. The well-mixed variant draws from the whole
population; the graph variant draws only from neighbours in a fixed topology.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _sample_rows(weights: np.ndarray, types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one column per row with probability proportional to ``weights``."""
    totals = weights.sum(axis=1)
    cdf = np.cumsum(weights, axis=1)
    u = rng.random(weights.shape[0]) * totals
    new = (cdf <= u[:, None]).sum(axis=1)
    new = np.minimum(new, weights.shape[1] - 1)
    empty = totals <= 0.0
    new[empty] = types[empty]
    return new


def _sample_neighbours(n: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform ``M`` distinct neighbours per row, excluding the row itself.

    The indices of the ``M`` smallest i.i.d. keys form a uniform ``M``-subset;
    setting the diagonal to infinity removes the focal individual exactly.
    """
    keys = rng.random((n, n))
    keys[np.arange(n), np.arange(n)] = np.inf
    return np.argpartition(keys, M - 1, axis=1)[:, :M]


def well_mixed_voter(
    types: np.ndarray,
    rewards: np.ndarray,
    rng: np.random.Generator,
    n_types: int | None = None,
    M: int | None = None,
) -> np.ndarray:
    """One weighted-voter step in a well-mixed population.

    ``M`` is the neighbourhood size. ``None`` (or ``M >= N - 1``) means the
    whole population, with the focal individual excluded exactly as in the
    paper's Eq. (B8). A finite ``M`` draws ``M`` distinct neighbours uniformly
    without replacement, again excluding the focal individual.
    """
    types = np.asarray(types, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    n = types.size
    if n_types is None:
        n_types = int(types.max()) + 1

    if M is None or M >= n - 1:
        global_votes = np.bincount(types, weights=rewards, minlength=n_types)
        onehot = np.zeros((n, n_types))
        onehot[np.arange(n), types] = 1.0
        weights = np.tile(global_votes, (n, 1)) - rewards[:, None] * onehot
        weights = np.clip(weights, 0.0, None)
    else:
        if M < 1:
            raise ValueError("M must be at least 1")
        idx = _sample_neighbours(n, M, rng)
        weights = np.zeros((n, n_types))
        np.add.at(
            weights,
            (np.repeat(np.arange(n), M), types[idx].ravel()),
            rewards[idx].ravel(),
        )

    return _sample_rows(weights, types, rng)


def graph_voter(
    types: np.ndarray,
    rewards: np.ndarray,
    adj: sparse.spmatrix,
    rng: np.random.Generator,
    n_types: int | None = None,
) -> np.ndarray:
    """One weighted-voter step on a fixed communication graph."""
    types = np.asarray(types, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    n = types.size
    if n_types is None:
        n_types = int(types.max()) + 1

    onehot = np.zeros((n, n_types))
    onehot[np.arange(n), types] = 1.0
    weights = np.asarray(adj @ (onehot * rewards[:, None]), dtype=float)
    return _sample_rows(weights, types, rng)
