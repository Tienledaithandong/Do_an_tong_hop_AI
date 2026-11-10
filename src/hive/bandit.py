"""Multi-armed bandit environment with uniformly noisy rewards.

Mirrors the environment of Soma et al. (2026), Appendix B: reward means ``q_a``
are equally spaced over a scenario range and each pull returns
``U(q_a - delta, q_a + delta)`` clipped to [0, 1].
"""

from __future__ import annotations

import numpy as np

SCENARIOS: dict[str, tuple[float, float]] = {
    "low": (0.1, 0.4),
    "middle": (0.4, 0.7),
    "high": (0.6, 0.9),
}


def make_q(scenario: str = "middle", n: int = 10) -> np.ndarray:
    """Return ``n`` equally spaced reward means inside the scenario range."""
    lo, hi = SCENARIOS[scenario]
    return np.linspace(lo, hi, n)


def expected_rewards(q: np.ndarray, delta: float) -> np.ndarray:
    """Mean of the *clipped* reward ``E[clip(U(q-delta, q+delta), 0, 1)]``.

    When ``q +/- delta`` leaves the unit interval the clipping biases the mean
    away from ``q``, so a replicator reference driven by raw ``q`` would compare
    against a different environment than the one the learners actually face.
    """
    q = np.asarray(q, dtype=float)
    if delta <= 0:
        return np.clip(q, 0.0, 1.0)
    lo = q - delta
    hi = q + delta
    width = hi - lo
    a = np.maximum(lo, 0.0)
    b = np.minimum(hi, 1.0)
    mid_mass = np.clip(b - a, 0.0, None) / width
    mid_mean = np.where(b > a, (a + b) / 2.0, 0.0)
    p_above = np.clip(hi - 1.0, 0.0, None) / width
    return mid_mass * mid_mean + p_above


def sample_rewards(
    q: np.ndarray, actions: np.ndarray, delta: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw one noisy reward per action, clipped to the unit interval."""
    q = np.asarray(q, dtype=float)
    actions = np.asarray(actions, dtype=int)
    low = q[actions] - delta
    high = q[actions] + delta
    return np.clip(rng.uniform(low, high), 0.0, 1.0)


class Bandit:
    """Stateless bandit; all arms are pulled independently every step."""

    def __init__(
        self,
        q: np.ndarray,
        delta: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.q = np.asarray(q, dtype=float)
        self.delta = float(delta)
        self.rng = np.random.default_rng() if rng is None else rng
        self.n = int(self.q.size)

    @property
    def best(self) -> int:
        """Index of the best arm under the hidden reward means."""
        return int(np.argmax(self.q))

    def pull(self, actions: np.ndarray) -> np.ndarray:
        """Rewards for a batch of actions."""
        return sample_rewards(self.q, actions, self.delta, self.rng)
