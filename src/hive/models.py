"""Policy update rules from the HiveMindRL paper.

``trd_step`` / ``mrd_step`` are the Taylor and Maynard-Smith replicator
dynamics, used as the analytical reference. ``p_cl_step`` / ``p_mcl_step`` are
the parallel Cross Learning and Maynard-Cross Learning rules, which use only
sampled actions and rewards.
"""

from __future__ import annotations

import numpy as np

from .bandit import sample_rewards


def _project(pi: np.ndarray) -> np.ndarray:
    """Clamp to [0, 1] and renormalise so the policy stays a distribution."""
    pi = np.clip(np.asarray(pi, dtype=float), 0.0, 1.0)
    total = pi.sum()
    if total <= 0.0:
        return np.full(pi.size, 1.0 / pi.size)
    return pi / total


def trd_step(pi: np.ndarray, q: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Euler step of the Taylor replicator dynamics."""
    pi = np.asarray(pi, dtype=float)
    q = np.asarray(q, dtype=float)
    value = float(pi @ q)
    return _project(pi + dt * pi * (q - value))


def mrd_step(pi: np.ndarray, q: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Euler step of the Maynard-Smith replicator dynamics."""
    pi = np.asarray(pi, dtype=float)
    q = np.asarray(q, dtype=float)
    value = float(pi @ q)
    if value <= 1e-12:
        return pi.copy()
    return _project(pi + dt * pi * (q - value) / value)


def p_cl_step(
    pi: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
    B: int = 1000,
    delta: float = 0.1,
) -> np.ndarray:
    """One parallel Cross Learning update from ``B`` sampled actions."""
    pi = np.asarray(pi, dtype=float)
    n = pi.size
    actions = rng.choice(n, size=B, p=pi)
    rewards = sample_rewards(q, actions, delta, rng)
    counts = np.bincount(actions, weights=rewards, minlength=n)
    return _project(pi + (counts - pi * rewards.sum()) / B)


def p_mcl_step(
    pi: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
    B: int = 1000,
    delta: float = 0.1,
) -> np.ndarray:
    """One parallel Maynard-Cross Learning update from ``B`` sampled actions.

    Algebraically the update collapses to the reward-weighted action frequency
    ``pi'_a = sum_{i: a_i = a} r_i / sum_i r_i``, which is exactly the weighted
    voter update of the population model.
    """
    pi = np.asarray(pi, dtype=float)
    n = pi.size
    actions = rng.choice(n, size=B, p=pi)
    rewards = sample_rewards(q, actions, delta, rng)
    total = rewards.sum()
    if total <= 1e-12:
        return pi.copy()
    counts = np.bincount(actions, weights=rewards, minlength=n)
    return _project(counts / total)
