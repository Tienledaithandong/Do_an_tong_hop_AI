"""Experiment B: how the communication graph shapes collective learning.

The swarm is placed on a fixed graph. Every individual copies the type of a
neighbour drawn proportionally to that neighbour's reward. The question is how
the graph's relaxation time ``tau`` maps onto the time the swarm needs to agree
on the best option.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .bandit import Bandit, make_q
from .graphs import average_path_length, degree_stats, relaxation_time, rgg_graph, ws_graph
from .metrics import best_mass, consensus, l2_to_reference, t90
from .models import mrd_step
from .swarm import graph_voter, well_mixed_voter

WS_P: list[float] = [0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
RGG_C: list[float] = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]


def _initial_types(rng: np.random.Generator, n: int, N: int) -> np.ndarray:
    types = np.repeat(np.arange(n), N // n)
    if types.size < N:
        types = np.concatenate([types, rng.integers(0, n, N - types.size)])
    rng.shuffle(types)
    return types


def reference_mrd(q: np.ndarray, steps: int) -> np.ndarray:
    """Deterministic MRD curve from a uniform policy, the well-mixed benchmark."""
    pi = np.full(q.size, 1.0 / q.size)
    best = int(np.argmax(q))
    curve = np.empty(steps + 1)
    curve[0] = pi[best]
    for step in range(steps):
        pi = mrd_step(pi, q)
        curve[step + 1] = pi[best]
    return curve


def simulate_on_graph(
    adj: sparse.spmatrix,
    q: np.ndarray,
    steps: int = 400,
    seeds: int = 100,
    delta: float = 0.1,
    n_types: int | None = None,
    mode: str = "graph",
) -> dict[str, np.ndarray]:
    """Run the weighted voter on a fixed topology and summarise every seed."""
    q = np.asarray(q, dtype=float)
    n = int(q.size) if n_types is None else n_types
    best = int(np.argmax(q))
    N = adj.shape[0]
    mrd_curve = reference_mrd(q, steps)

    curves = np.zeros((seeds, steps + 1))
    t90s = np.zeros(seeds)
    hits = np.zeros(seeds, dtype=bool)
    finals = np.zeros(seeds)
    consensuses = np.zeros(seeds)
    l2s = np.zeros(seeds)
    tail = max(1, steps // 10)

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        bandit = Bandit(q, delta=delta, rng=rng)
        types = _initial_types(rng, n, N)
        curve = curves[seed]
        curve[0] = best_mass(types, best)

        for step in range(steps):
            rewards = bandit.pull(types)
            if mode == "well_mixed":
                types = well_mixed_voter(types, rewards, rng, n_types=n)
            else:
                types = graph_voter(types, rewards, adj, rng, n_types=n)
            curve[step + 1] = best_mass(types, best)

        t90s[seed], hits[seed] = t90(curve)
        finals[seed] = float(curve[-tail:].mean())
        counts = np.bincount(types, minlength=n) / N
        consensuses[seed] = consensus(counts)
        l2s[seed] = l2_to_reference(curve, mrd_curve)

    return {
        "curves": curves,
        "t90": t90s,
        "hit": hits,
        "final": finals,
        "consensus": consensuses,
        "l2_mrd": l2s,
        "mrd": mrd_curve,
    }


def run_graph_sweep(
    N: int = 500,
    steps: int = 400,
    seeds: int = 100,
    n: int = 10,
    scenario: str = "middle",
    delta: float = 0.1,
    ws_p: list[float] | None = None,
    rgg_c: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Sweep Watts-Strogatz rewiring and RGG radius; return per-seed rows and mean curves."""
    ws_p = WS_P if ws_p is None else ws_p
    rgg_c = RGG_C if rgg_c is None else rgg_c
    q = make_q(scenario, n)

    dens = np.ones((N, N)) - np.eye(N)
    configs: list[tuple[str, float, sparse.spmatrix, str]] = [
        ("complete", 1.0, sparse.csr_matrix(dens), "well_mixed")
    ]
    for p in ws_p:
        configs.append(("ws", float(p), ws_graph(N, 6, p, seed=0), "graph"))
    rc = float(np.sqrt(np.log(N) / (np.pi * N)))
    for c in rgg_c:
        configs.append(("rgg", float(c), rgg_graph(N, c * rc, seed=0), "graph"))

    rows: list[dict[str, float | str | int]] = []
    mean_curves: dict[str, np.ndarray] = {}

    for family, param, adj, mode in configs:
        lambda2, tau = relaxation_time(adj)
        stats = degree_stats(adj)
        path_length = average_path_length(adj)
        res = simulate_on_graph(
            adj, q, steps=steps, seeds=seeds, delta=delta, n_types=n, mode=mode
        )
        key = f"{family}:{param:g}"
        mean_curves[key] = res["curves"].mean(axis=0)

        for seed in range(seeds):
            rows.append(
                {
                    "family": family,
                    "param": param,
                    "seed": seed,
                    "n": N,
                    "lambda2": lambda2,
                    "tau": tau,
                    "path_length": path_length,
                    "degree_mean": stats["mean"],
                    "degree_cv": stats["cv"],
                    "t90": res["t90"][seed],
                    "t90_hit": bool(res["hit"][seed]),
                    "final_best_mass": res["final"][seed],
                    "consensus": res["consensus"][seed],
                    "l2_mrd": res["l2_mrd"][seed],
                }
            )

    return pd.DataFrame(rows), mean_curves
