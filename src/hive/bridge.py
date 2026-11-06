"""Replicated source-room experiment with a bridge-capacity decomposition.

The single-graph pilot in ``check_two_room.py`` showed that bridge width
changes the time the best type needs to reach the target room even though the
mean path length ``L`` barely moves. This module repeats that experiment over
several graphs and room sizes, and splits the delay into two measurable parts:

* ``t_first``  - first step at which the target room holds any copy of the
  best type (the wait *at the bridge*);
* ``spread50`` / ``spread90`` - time from first arrival to 50% / 90% of the
  target room (the spread *inside* the room).

It also computes the bridge conductance ``phi = bridge / (N / 2)`` so that
candidate predictors can be compared on the same data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .bandit import make_q
from .graph_sweep import simulate_two_room_source
from .graphs import average_path_length, relaxation_time, two_room_graph


def first_arrival(target_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First step with a non-zero best-type mass in the target room.

    Returns ``(step, hit)``. Unreached seeds get ``step = horizon + 1`` and
    ``hit = False`` so they are never mistaken for a real measurement.
    """
    curves = np.asarray(target_curves, dtype=float)
    reached = curves > 0.0
    hit = reached.any(axis=1)
    step = np.where(hit, reached.argmax(axis=1), curves.shape[1]).astype(float)
    return step, hit


def arrival_events(target_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count arrivals (0 -> >0) and extinctions (>0 -> 0) per seed.

    Every extinction is an invasion that the target room's resident types
    pushed back out, which a single "first arrival" time cannot show.
    """
    present = np.asarray(target_curves, dtype=float) > 0.0
    prev, curr = present[:, :-1], present[:, 1:]
    arrivals = (~prev & curr).sum(axis=1)
    extinctions = (prev & ~curr).sum(axis=1)
    return arrivals, extinctions


def run_replicates(
    sizes: dict[int, int],
    bridges: list[int],
    graph_seeds: int = 4,
    seeds: int = 20,
    n_types: int = 10,
    scenario: str = "middle",
    delta: float = 0.1,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run the source-room experiment on a grid; return one row per simulation seed.

    ``sizes`` maps the population size ``N`` to the number of steps to simulate.
    Each ``(N, bridge, graph_seed)`` uses its own random graph, so variation
    across graph seeds is part of the reported uncertainty.
    """
    q = make_q(scenario, n_types)
    rows: list[dict[str, float | int | bool]] = []
    for N, steps in sizes.items():
        for bridge in bridges:
            for graph_seed in range(graph_seeds):
                adj = two_room_graph(N, bridge=int(bridge), seed=graph_seed)
                _, tau = relaxation_time(adj)
                path_length = average_path_length(adj)
                res = simulate_two_room_source(
                    adj, q, steps=steps, seeds=seeds, delta=delta, n_types=n_types
                )
                t_first, hit_first = first_arrival(res["target_curves"])
                arrivals, extinctions = arrival_events(res["target_curves"])
                for seed in range(seeds):
                    rows.append(
                        {
                            "N": N,
                            "steps": steps,
                            "bridge": int(bridge),
                            "graph_seed": graph_seed,
                            "seed": seed,
                            "tau": tau,
                            "path_length": path_length,
                            "phi": bridge / (N / 2),
                            "t_first": t_first[seed],
                            "hit_first": bool(hit_first[seed]),
                            "t50": res["t50_target"][seed],
                            "hit50": bool(res["hit50_target"][seed]),
                            "t90": res["t90_target"][seed],
                            "hit90": bool(res["hit90_target"][seed]),
                            "final_target": res["final_target"][seed],
                            "arrivals": int(arrivals[seed]),
                            "extinctions": int(extinctions[seed]),
                        }
                    )
                if progress is not None:
                    progress(f"N={N} bridge={bridge} graph_seed={graph_seed}")
    return pd.DataFrame(rows)


def censored_median(values: np.ndarray, hit: np.ndarray) -> float:
    """Median over *all* seeds with unreached seeds treated as ``+inf``.

    The result is ``inf`` when fewer than half of the seeds reached the
    threshold, which is the honest answer: the median is then not observed.
    """
    values = np.where(np.asarray(hit, dtype=bool), np.asarray(values, dtype=float), np.inf)
    return float(np.median(values))


def bootstrap_ci(
    values: np.ndarray,
    hit: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    cluster: np.ndarray | None = None,
) -> tuple[float, float]:
    """95% bootstrap CI for the censored median.

    With ``cluster`` (e.g. the graph seed) whole clusters are resampled, so the
    interval reflects graph-to-graph variation rather than only seed noise.
    """
    values = np.asarray(values, dtype=float)
    hit = np.asarray(hit, dtype=bool)
    rng = np.random.default_rng(seed)
    if cluster is None:
        idx = rng.integers(0, values.size, size=(n_boot, values.size))
    else:
        cluster = np.asarray(cluster)
        groups = [np.flatnonzero(cluster == c) for c in np.unique(cluster)]
        idx_rows = []
        for _ in range(n_boot):
            picked = rng.integers(0, len(groups), size=len(groups))
            idx_rows.append(np.concatenate([groups[g] for g in picked]))
        idx = np.stack(idx_rows)
    meds = np.array(
        [
            np.median(np.where(hit[i], values[i], np.inf))
            for i in idx
        ]
    )
    # method="nearest" keeps +inf (censored medians) from turning into NaN.
    lo = float(np.percentile(meds, 2.5, method="nearest"))
    hi = float(np.percentile(meds, 97.5, method="nearest"))
    return lo, hi


def summarise(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """One row per ``(N, bridge)`` with censored medians and graph-cluster CIs."""
    out = []
    for (N, bridge), g in df.groupby(["N", "bridge"]):
        row: dict[str, float | int] = {
            "N": int(N),
            "bridge": int(bridge),
            "phi": float(g["phi"].iloc[0]),
            "tau": float(g["tau"].mean()),
            "path_length": float(g["path_length"].mean()),
            "runs": int(len(g)),
            "graphs": int(g["graph_seed"].nunique()),
        }
        for key, hit in (("t_first", "hit_first"), ("t50", "hit50"), ("t90", "hit90")):
            row[f"{key}_median"] = censored_median(g[key], g[hit])
            lo, hi = bootstrap_ci(
                g[key].to_numpy(), g[hit].to_numpy(), n_boot=n_boot, cluster=g["graph_seed"].to_numpy()
            )
            row[f"{key}_lo"] = lo
            row[f"{key}_hi"] = hi
            row[f"{key}_hit_rate"] = float(g[hit].mean())
        row["final_target95"] = float((g["final_target"] >= 0.95).mean())
        row["arrivals_mean"] = float(g["arrivals"].mean())
        row["extinctions_mean"] = float(g["extinctions"].mean())
        out.append(row)
    return pd.DataFrame(out).sort_values(["N", "bridge"]).reset_index(drop=True)


def bootstrap_exponent(
    df: pd.DataFrame,
    key: str,
    hit: str,
    n_boot: int = 300,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI for the log-log slope of ``key`` against ``phi``.

    Resamples graph seeds with replacement inside every ``(N, bridge)`` cell,
    recomputes the censored medians and refits. Cells whose median is not
    observed (``inf``) are dropped from that replicate.
    """
    rng = np.random.default_rng(seed)
    cells = []
    for (_, _), g in df.groupby(["N", "bridge"]):
        by_graph = [grp for _, grp in g.groupby("graph_seed")]
        cells.append((float(g["phi"].iloc[0]), by_graph))
    slopes = []
    for _ in range(n_boot):
        xs, ys = [], []
        for phi, by_graph in cells:
            picked = rng.integers(0, len(by_graph), size=len(by_graph))
            part = pd.concat([by_graph[i] for i in picked])
            med = censored_median(part[key], part[hit])
            if np.isfinite(med):
                xs.append(phi)
                ys.append(med)
        if len(xs) >= 3:
            slopes.append(fit_loglog(np.array(xs), np.array(ys))["slope"])
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def fit_loglog2(
    bridge: np.ndarray, size: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Fit ``log y = a + b log(bridge) + c log(N)``.

    If the wait depends only on the conductance ``phi = 2 * bridge / N`` then
    ``c = -b``; a clear gap between them means room size matters beyond phi.
    """
    X = np.column_stack(
        [np.ones(len(y)), np.log(np.asarray(bridge, float)), np.log(np.asarray(size, float))]
    )
    ly = np.log(np.asarray(y, dtype=float))
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ coef
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {
        "intercept": float(coef[0]),
        "bridge_exp": float(coef[1]),
        "size_exp": float(coef[2]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def fit_loglog(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least squares ``log y = a + b log x``; returns slope, intercept and R^2."""
    x = np.log(np.asarray(x, dtype=float))
    y = np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }
