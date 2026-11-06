"""Source-room experiment on two rooms with uneven degrees.

With clique or regular rooms every bridge endpoint has the same degree, so
"bridge edges per room neighbour" (``bridge / degree``) cannot be told apart
from anything that depends on the endpoints themselves. Here the rooms are
Barabasi-Albert graphs and the bridge is deliberately attached to hubs, leaves
or random nodes of either room.

Two candidate predictors of the wait for the best type to reach the target:

* ``phi_deg_room = bridge / mean degree of the target room`` - the room
  average, identical for every placement;
* ``inv_end_target = sum over bridge edges of 1 / deg(target endpoint)`` -
  a target endpoint hears its one cross-bridge neighbour with weight about
  ``1 / deg``, so this is the (unnormalised) per-step invasion rate.

If the endpoint predictor wins, placement must matter on the target side and
much less on the source side.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from .bandit import make_q
from .bridge import (
    arrival_events,
    bootstrap_ci,
    censored_median,
    first_arrival,
    fit_loglog,
    fit_loglog_multi,
)
from .graph_sweep import simulate_two_room_source
from .graphs import average_path_length, relaxation_time, two_room_hetero


def endpoint_stats(adj: sparse.spmatrix) -> dict[str, float]:
    """Degree statistics of the bridge endpoints and of the target room.

    Degrees include the bridge edge itself, which is what a node actually
    sees when it samples a neighbour.
    """
    adj = sparse.csr_matrix(adj)
    n = adj.shape[0]
    half = n // 2
    deg = np.asarray(adj.sum(axis=1)).ravel()
    cross = adj[:half, half:].tocoo()
    src = cross.row
    tgt = cross.col + half
    return {
        "bridge_edges": float(src.size),
        "inv_end_target": float((1.0 / deg[tgt]).sum()),
        "inv_end_source": float((1.0 / deg[src]).sum()),
        "end_deg_target": float(deg[tgt].mean()),
        "end_deg_source": float(deg[src].mean()),
        "mean_deg_target": float(deg[half:].mean()),
        "mean_deg_source": float(deg[:half].mean()),
        "max_deg_target": float(deg[half:].max()),
    }


def run_hetero(
    N: int,
    steps: int,
    bridges: list[int],
    layouts: list[dict],
    graph_seeds: int = 5,
    seeds: int = 20,
    n_types: int = 10,
    scenario: str = "middle",
    delta: float = 0.1,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run every ``layout`` x ``bridge`` x ``graph_seed``; one row per seed.

    A layout is ``{"kind": "ba" | "regular", "m": int, "source": str,
    "target": str}``; its string ``name`` is stored as ``layout``.
    """
    q = make_q(scenario, n_types)
    rows: list[dict[str, float | int | bool | str]] = []
    for layout in layouts:
        name = layout_name(layout)
        for bridge in bridges:
            for graph_seed in range(graph_seeds):
                adj = two_room_hetero(
                    N,
                    bridge=int(bridge),
                    seed=graph_seed,
                    kind=layout["kind"],
                    m=layout["m"],
                    source=layout["source"],
                    target=layout["target"],
                )
                stats = endpoint_stats(adj)
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
                            "layout": name,
                            "kind": layout["kind"],
                            "source": layout["source"],
                            "target": layout["target"],
                            "N": N,
                            "steps": steps,
                            "bridge": int(bridge),
                            "graph_seed": graph_seed,
                            "seed": seed,
                            "tau": tau,
                            "path_length": path_length,
                            **stats,
                            "t_first": t_first[seed],
                            "hit_first": bool(hit_first[seed]),
                            "t50": res["t50_target"][seed],
                            "hit50": bool(res["hit50_target"][seed]),
                            "final_target": res["final_target"][seed],
                            "arrivals": int(arrivals[seed]),
                            "extinctions": int(extinctions[seed]),
                        }
                    )
                if progress is not None:
                    progress(f"{name} bridge={bridge} graph_seed={graph_seed}")
    df = pd.DataFrame(rows)
    df["phi_deg_room"] = df["bridge_edges"] / df["mean_deg_target"]
    return df


def layout_name(layout: dict) -> str:
    base = f"{layout['kind']}{layout['m']}"
    return f"{base}:{layout['source']}->{layout['target']}"


CELL = ["layout", "bridge"]
PREDICTORS = ["phi_deg_room", "inv_end_target", "inv_end_source", "tau"]


def summarise_hetero(df: pd.DataFrame, n_boot: int = 800) -> pd.DataFrame:
    """One row per ``(layout, bridge)`` with censored medians and graph-cluster CIs."""
    out = []
    for (layout, bridge), g in df.groupby(CELL):
        row: dict[str, float | int | str] = {
            "layout": layout,
            "kind": g["kind"].iloc[0],
            "source": g["source"].iloc[0],
            "target": g["target"].iloc[0],
            "bridge": int(bridge),
            "runs": int(len(g)),
            "graphs": int(g["graph_seed"].nunique()),
        }
        for col in [
            "phi_deg_room",
            "inv_end_target",
            "inv_end_source",
            "end_deg_target",
            "end_deg_source",
            "mean_deg_target",
            "tau",
            "path_length",
        ]:
            row[col] = float(g[col].mean())
        for key, hit in (("t_first", "hit_first"), ("t50", "hit50")):
            row[f"{key}_median"] = censored_median(g[key], g[hit])
            lo, hi = bootstrap_ci(
                g[key].to_numpy(),
                g[hit].to_numpy(),
                n_boot=n_boot,
                cluster=g["graph_seed"].to_numpy(),
            )
            row[f"{key}_lo"] = lo
            row[f"{key}_hi"] = hi
            row[f"{key}_hit_rate"] = float(g[hit].mean())
        row["arrivals_mean"] = float(g["arrivals"].mean())
        row["extinctions_mean"] = float(g["extinctions"].mean())
        out.append(row)
    return pd.DataFrame(out).sort_values(["layout", "bridge"]).reset_index(drop=True)


def compare_hetero_predictors(summary: pd.DataFrame, key: str = "t_first_median") -> dict:
    """Single and joint log-log fits of ``key`` over the finite cells."""
    ok = summary[np.isfinite(summary[key])]
    fits: dict = {
        "cells": int(len(ok)),
        "single": {p: fit_loglog(ok[p], ok[key]) for p in PREDICTORS},
    }
    fits["target_and_source"] = fit_loglog_multi(
        {"inv_end_target": ok["inv_end_target"], "inv_end_source": ok["inv_end_source"]},
        ok[key],
    )
    fits["target_and_room"] = fit_loglog_multi(
        {"inv_end_target": ok["inv_end_target"], "phi_deg_room": ok["phi_deg_room"]},
        ok[key],
    )
    fits["target_and_tau"] = fit_loglog_multi(
        {"inv_end_target": ok["inv_end_target"], "tau": ok["tau"]}, ok[key]
    )
    return fits


def placement_effects(summary: pd.DataFrame, key: str = "t_first_median") -> pd.DataFrame:
    """Median ratio of ``key`` to the random/random layout at each bridge width.

    Values below 1 mean the placement speeds up the first arrival. Only BA
    layouts are used; the table is indexed by ``(source, target)``.
    """
    ba = summary[summary["kind"] == "ba"]
    base = ba[(ba["source"] == "random") & (ba["target"] == "random")].set_index("bridge")
    rows = []
    for (source, target), g in ba.groupby(["source", "target"]):
        g = g.set_index("bridge")
        num = g[key].to_numpy(dtype=float)
        den = base[key].reindex(g.index).to_numpy(dtype=float)
        valid = np.isfinite(num) & np.isfinite(den) & (den > 0)
        ratio = np.where(valid, num / np.where(den > 0, den, 1.0), np.nan)
        if not valid.any():
            rows.append({"source": source, "target": target, "ratio_median": np.nan,
                         "ratio_min": np.nan, "ratio_max": np.nan, "cells": 0})
            continue
        rows.append(
            {
                "source": source,
                "target": target,
                "ratio_median": float(np.nanmedian(ratio)),
                "ratio_min": float(np.nanmin(ratio)),
                "ratio_max": float(np.nanmax(ratio)),
                "cells": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_hetero(
    df: pd.DataFrame,
    key: str,
    hit: str,
    predictors: list[str],
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, list[float]]:
    """95% CIs for the exponents of ``log key = a + sum_k b_k log predictor_k``.

    Whole graphs are resampled inside every ``(layout, bridge)`` cell and the
    per-cell predictors are recomputed as means over the resampled graphs.
    """
    rng = np.random.default_rng(seed)
    cells = [
        [grp for _, grp in g.groupby("graph_seed")] for _, g in df.groupby(CELL)
    ]
    draws: dict[str, list[float]] = {p: [] for p in predictors}
    for _ in range(n_boot):
        xs: dict[str, list[float]] = {p: [] for p in predictors}
        ys: list[float] = []
        for by_graph in cells:
            picked = rng.integers(0, len(by_graph), size=len(by_graph))
            part = pd.concat([by_graph[i] for i in picked])
            med = censored_median(part[key], part[hit])
            if np.isfinite(med):
                ys.append(med)
                for p in predictors:
                    xs[p].append(float(part[p].mean()))
        if len(ys) > len(predictors) + 1:
            fit = fit_loglog_multi({p: np.array(v) for p, v in xs.items()}, np.array(ys))
            for p in predictors:
                draws[p].append(fit[f"{p}_exp"])
    return {
        p: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for p, v in draws.items()
        if v
    }
