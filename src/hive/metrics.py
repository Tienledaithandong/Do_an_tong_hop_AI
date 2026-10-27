"""Metrics for learning curves of a swarm or a policy."""

from __future__ import annotations

import numpy as np


def best_mass(types: np.ndarray, best: int) -> float:
    """Fraction of individuals holding the optimal type."""
    types = np.asarray(types)
    return float(np.mean(types == best))


def _smooth(curve: np.ndarray, window: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    if window <= 1 or curve.size == 0:
        return curve
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode="same")


def t90(
    curve: np.ndarray, threshold: float = 0.9, window: int = 5
) -> tuple[float, bool]:
    """First step where the smoothed curve reaches ``threshold``.

    Returns ``(step, hit)``. When the threshold is never reached the horizon is
    returned with ``hit=False``, so callers can report censoring instead of
    silently averaging a fake value.
    """
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return 0.0, False
    smoothed = _smooth(curve, window)
    hits = np.flatnonzero(smoothed >= threshold)
    if hits.size == 0:
        return float(curve.size), False
    return float(hits[0]), True


def consensus(pi: np.ndarray) -> float:
    """Mass on the most popular option."""
    pi = np.asarray(pi, dtype=float)
    return float(pi.max()) if pi.size else 0.0


def l2_to_reference(curve: np.ndarray, reference: np.ndarray) -> float:
    """Root-mean-square distance between two learning curves."""
    curve = np.asarray(curve, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n = min(curve.size, reference.size)
    return float(np.sqrt(np.mean((curve[:n] - reference[:n]) ** 2)))
