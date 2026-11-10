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
    # Pad with the edge value, not zeros: zero-padding drags the smoothed
    # curve down near the horizon and turns late crossings into censored runs.
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(curve, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def threshold_time(
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


def t90(
    curve: np.ndarray, threshold: float = 0.9, window: int = 5
) -> tuple[float, bool]:
    """First step where the smoothed curve reaches 90% of the best option."""
    return threshold_time(curve, threshold=threshold, window=window)


def kaplan_meier(
    times: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan–Meier survival curve for right-censored threshold times.

    ``events[i]`` is True when the threshold was reached at ``times[i]`` and
    False when the run was censored at the horizon. Returns knot times
    (starting at 0) and survival probabilities. Tied event times are grouped;
    censored runs only shrink the risk set, and a run censored at an event
    time is still counted at risk at that time (standard convention).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if times.shape != events.shape:
        raise ValueError("times and events must have the same shape")
    if times.size == 0:
        return np.empty(0), np.empty(0)
    knots_t = [0.0]
    knots_s = [1.0]
    surv = 1.0
    for t in np.unique(times):
        d = int(np.sum(events & (times == t)))
        if d:
            at_risk = int(np.sum(times >= t))
            surv *= 1.0 - d / at_risk
            knots_t.append(float(t))
            knots_s.append(surv)
    return np.asarray(knots_t), np.asarray(knots_s)


def rmst(times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    """Restricted mean survival time: area under the KM curve up to ``horizon``.

    Unlike a median over hit-only runs, this stays comparable when hit rates
    differ across conditions: a censored run contributes the full time it was
    observed instead of being dropped.
    """
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return float("nan")
    ts, ss = kaplan_meier(times, events)
    keep = ts <= horizon
    ts = np.append(ts[keep], horizon)
    ss = np.append(ss[keep], ss[keep][-1])
    return float(np.sum(np.diff(ts) * ss[:-1]))


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
