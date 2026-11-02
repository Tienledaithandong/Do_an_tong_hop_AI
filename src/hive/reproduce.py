"""Experiment A: reproduce the well-mixed learning curves of the paper.

Three learners start from the same uniform policy and face the same bandit:
the weighted voter population, parallel Maynard-Cross Learning, and the
Maynard-Smith replicator dynamics used as the analytical reference.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .bandit import Bandit, make_q
from .metrics import best_mass
from .models import mrd_step, p_mcl_step
from .swarm import well_mixed_voter


def run_reproduction(
    n: int = 10,
    N: int = 1000,
    steps: int = 300,
    seeds: int = 100,
    scenario: str = "middle",
    delta: float = 0.1,
    B: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run the well-mixed comparison and return curves of shape (seeds, steps+1).

    ``progress`` is called as ``progress(completed_seeds, total_seeds)`` after
    each seed so a UI can show a progress bar.
    """
    q = make_q(scenario, n)
    best = int(np.argmax(q))
    batch = N if B is None else B

    curves = {
        "well_mixed": np.zeros((seeds, steps + 1)),
        "pmcl": np.zeros((seeds, steps + 1)),
        "mrd": np.zeros((seeds, steps + 1)),
    }

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        bandit = Bandit(q, delta=delta, rng=rng)

        types = np.repeat(np.arange(n), N // n)
        if types.size < N:
            types = np.concatenate([types, rng.integers(0, n, N - types.size)])
        rng.shuffle(types)

        pi_pmcl = np.full(n, 1.0 / n)
        pi_mrd = np.full(n, 1.0 / n)
        curves["well_mixed"][seed, 0] = best_mass(types, best)
        curves["pmcl"][seed, 0] = pi_pmcl[best]
        curves["mrd"][seed, 0] = pi_mrd[best]

        for step in range(steps):
            rewards = bandit.pull(types)
            types = well_mixed_voter(types, rewards, rng, n_types=n)
            curves["well_mixed"][seed, step + 1] = best_mass(types, best)

            pi_pmcl = p_mcl_step(pi_pmcl, q, rng, B=batch, delta=delta)
            curves["pmcl"][seed, step + 1] = pi_pmcl[best]

            pi_mrd = mrd_step(pi_mrd, q)
            curves["mrd"][seed, step + 1] = pi_mrd[best]

        if progress is not None:
            progress(seed + 1, seeds)

    return curves
