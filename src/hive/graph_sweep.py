"""Experiment B: how the communication graph shapes collective learning.

The swarm is placed on a fixed graph. Every individual copies the type of a
neighbour drawn proportionally to that neighbour's reward. The question is how
the graph's relaxation time ``tau`` maps onto the time the swarm needs to agree
on the best option.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from .bandit import Bandit, make_q
from .graphs import (
    average_path_length,
    degree_stats,
    relaxation_time,
    rgg_graph,
    two_room_graph,
    ws_graph,
)
from .metrics import best_mass, consensus, l2_to_reference, t90, threshold_time
from .models import mrd_step
from .swarm import graph_voter, well_mixed_voter

from hive.config import WS_P, RGG_C, BRIDGE_P


def _initial_types(rng: np.random.Generator, n: int, N: int) -> np.ndarray:
    types = np.repeat(np.arange(n), N // n)
    if types.size < N:
        types = np.concatenate([types, rng.integers(0, n, N - types.size)])
    rng.shuffle(types)
    return types


def room_mask(N: int, room: int = 0) -> np.ndarray:
    """Boolean mask for one of the two rooms in ``two_room_graph``."""
    half = N // 2
    mask = np.arange(N) < half
    return mask if room == 0 else ~mask


def initial_types_source_room(
    rng: np.random.Generator,
    n: int,
    N: int,
    best: int,
    source_mask: np.ndarray,
) -> np.ndarray:
    """Balanced types with the optimal type confined to the source room.

    This is the intervention the original two-room experiment was missing:
    the target room starts with *no* copy of the best option, so the bridge
    is the only channel through which it can arrive.
    """
    target_mask = ~source_mask
    source_idx = np.flatnonzero(source_mask)
    target_idx = np.flatnonzero(target_mask)

    source_types = np.arange(source_idx.size) % n
    if best not in source_types:
        source_types[0] = best
    rng.shuffle(source_types)

    others = np.array([a for a in range(n) if a != best], dtype=int)
    target_types = others[np.arange(target_idx.size) % others.size]
    rng.shuffle(target_types)

    types = np.empty(N, dtype=int)
    types[source_idx] = source_types
    types[target_idx] = target_types
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
    progress: Callable[[int, int], None] | None = None,
    return_state: bool = False,
) -> dict[str, np.ndarray]:
    """Run the weighted voter on a fixed topology and summarise every seed.

    ``progress`` is called as ``progress(completed_seeds, total_seeds)`` after
    each seed, which lets a UI report progress without changing the numerics.
    ``return_state`` adds the final type vector of the last seed under
    ``"final_types"``, used for visualisation only.
    """
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
        if progress is not None:
            progress(seed + 1, seeds)

    result: dict[str, np.ndarray] = {
        "curves": curves,
        "t90": t90s,
        "hit": hits,
        "final": finals,
        "consensus": consensuses,
        "l2_mrd": l2s,
        "mrd": mrd_curve,
    }
    if return_state and seeds > 0:
        result["final_types"] = types.copy()
    return result


def simulate_two_room_source(
    adj: sparse.spmatrix,
    q: np.ndarray,
    steps: int = 600,
    seeds: int = 30,
    delta: float = 0.1,
    n_types: int | None = None,
    source_room: int = 0,
    progress: Callable[[int, int], None] | None = None,
    return_state: bool = False,
) -> dict[str, np.ndarray]:
    """Two-room intervention with the optimal type seeded in one room only.

    Both rooms in ``two_room_graph`` are cliques. With the usual balanced
    initial condition each room already contains the best type, so the bridge
    is irrelevant. Here the target room starts with only non-optimal types;
    ``t50_target`` and ``t90_target`` therefore measure how long information
    needs to cross the bridge, which is the quantity the original experiment
    claimed to isolate.
    """
    q = np.asarray(q, dtype=float)
    n = int(q.size) if n_types is None else n_types
    best = int(np.argmax(q))
    N = adj.shape[0]
    source_mask = room_mask(N, source_room)
    target_mask = ~source_mask
    mrd_curve = reference_mrd(q, steps)

    n_curves = ("curves", "source_curves", "target_curves")
    curves = {name: np.zeros((seeds, steps + 1)) for name in n_curves}
    arrays = {
        "t10_target": np.zeros(seeds),
        "t50_target": np.zeros(seeds),
        "t90_target": np.zeros(seeds),
        "t90": np.zeros(seeds),
        "final": np.zeros(seeds),
        "final_target": np.zeros(seeds),
        "consensus": np.zeros(seeds),
        "l2_mrd": np.zeros(seeds),
    }
    flags = {
        "hit10_target": np.zeros(seeds, dtype=bool),
        "hit50_target": np.zeros(seeds, dtype=bool),
        "hit90_target": np.zeros(seeds, dtype=bool),
        "hit": np.zeros(seeds, dtype=bool),
    }
    tail = max(1, steps // 10)

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        bandit = Bandit(q, delta=delta, rng=rng)
        types = initial_types_source_room(rng, n, N, best, source_mask)
        overall = curves["curves"][seed]
        source_curve = curves["source_curves"][seed]
        target_curve = curves["target_curves"][seed]
        overall[0] = best_mass(types, best)
        source_curve[0] = best_mass(types[source_mask], best)
        target_curve[0] = best_mass(types[target_mask], best)

        for step in range(steps):
            rewards = bandit.pull(types)
            types = graph_voter(types, rewards, adj, rng, n_types=n)
            overall[step + 1] = best_mass(types, best)
            source_curve[step + 1] = best_mass(types[source_mask], best)
            target_curve[step + 1] = best_mass(types[target_mask], best)

        for key, threshold, prefix in (
            ("t10_target", 0.10, "hit10_target"),
            ("t50_target", 0.50, "hit50_target"),
            ("t90_target", 0.90, "hit90_target"),
        ):
            arrays[key][seed], flags[prefix][seed] = threshold_time(
                target_curve, threshold=threshold
            )
        arrays["t90"][seed], flags["hit"][seed] = t90(overall)
        arrays["final"][seed] = float(overall[-tail:].mean())
        arrays["final_target"][seed] = float(target_curve[-tail:].mean())
        counts = np.bincount(types, minlength=n) / N
        arrays["consensus"][seed] = consensus(counts)
        arrays["l2_mrd"][seed] = l2_to_reference(overall, mrd_curve)
        if progress is not None:
            progress(seed + 1, seeds)

    result: dict[str, np.ndarray] = {
        **curves,
        **arrays,
        **flags,
        "mrd": mrd_curve,
        "source_mask": source_mask,
        "target_mask": target_mask,
    }
    if return_state and seeds > 0:
        result["final_types"] = types.copy()
    return result


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
    for b in BRIDGE_P:
        configs.append(("two_room", float(b), two_room_graph(N, bridge=int(b), seed=0), "graph"))

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
