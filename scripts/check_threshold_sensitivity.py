"""Sensitivity of threshold times to the crossing definition (review #2).

The headline ``t90`` reads the first crossing of a *centered* moving average,
which can anticipate the raw signal. This script re-runs the main law
configurations and the source-room pilot once, then recomputes every threshold
on the *identical* trajectories under three definitions: symmetric (centered),
causal (trailing) and raw. It reports how the fitted ``t90 ~ L`` line and the
two-room bridge effect move with the definition, so the report can state which
conclusions are robust to the measurement rule.

The law fit reuses ``config_summary`` + ``fit_law`` (the headline convention,
including its all-seed median), so the symmetric column reproduces the headline
line; the censoring-aware refit of the law is a separate concern (review #3).

Usage:
    python scripts/check_threshold_sensitivity.py
    python scripts/check_threshold_sensitivity.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.bandit import make_q
from hive.config import RUN_ALL, RGG_C, TWO_ROOM_SOURCE, WS_P
from hive.graph_sweep import simulate_on_graph, simulate_two_room_source
from hive.graphs import (
    average_path_length,
    degree_stats,
    relaxation_time,
    rgg_graph,
    two_room_graph,
    ws_graph,
)
from hive.metrics import THRESHOLD_METHODS, rmst, threshold_time
from hive.plots import config_summary, fit_law, plot_threshold_sensitivity
from hive.provenance import write_metadata

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def law_configs(N: int, ws_p: list[float], rgg_c: list[float]):
    dens = np.ones((N, N)) - np.eye(N)
    configs = [("complete", 1.0, sparse.csr_matrix(dens), "well_mixed")]
    for p in ws_p:
        configs.append(("ws", float(p), ws_graph(N, 6, p, seed=0), "graph"))
    rc = float(np.sqrt(np.log(N) / (np.pi * N)))
    for c in rgg_c:
        configs.append(("rgg", float(c), rgg_graph(N, c * rc, seed=0), "graph"))
    return configs


def conditional_median(times: np.ndarray, hit: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    hit = np.asarray(hit, dtype=bool)
    reached = times[hit]
    return float(np.median(reached)) if reached.size else float("nan")


def run_law(N, steps, seeds, n, scenario, delta, ws_p, rgg_c):
    q = make_q(scenario, n)
    rows = []
    for family, param, adj, mode in law_configs(N, ws_p, rgg_c):
        _, tau = relaxation_time(adj)
        path = average_path_length(adj)
        stats = degree_stats(adj)
        res = simulate_on_graph(
            adj, q, steps=steps, seeds=seeds, delta=delta, n_types=n, mode=mode
        )
        curves = res["curves"]
        crossings = {
            m: np.array([threshold_time(c, 0.9, 5, m) for c in curves])
            for m in THRESHOLD_METHODS
        }
        for seed in range(seeds):
            row = {
                "family": family,
                "param": param,
                "seed": seed,
                "tau": tau,
                "path_length": path,
                "degree_mean": stats["mean"],
                "degree_cv": stats["cv"],
                "final_best_mass": res["final"][seed],
                "consensus": res["consensus"][seed],
            }
            for m in THRESHOLD_METHODS:
                row[f"t90_{m}"] = crossings[m][seed, 0]
                row[f"hit_{m}"] = bool(crossings[m][seed, 1])
            rows.append(row)

    base = pd.DataFrame(rows)
    summaries, fits = {}, {}
    for m in THRESHOLD_METHODS:
        view = base.rename(columns={f"t90_{m}": "t90", f"hit_{m}": "t90_hit"})
        summary = config_summary(view)
        summaries[m] = summary
        intercept, slope, r2, n_train = fit_law(summary)
        fits[m] = {
            "intercept": float(intercept),
            "slope": float(slope),
            "r2": float(r2),
            "n_train": int(n_train),
        }
    law_table = pd.concat(
        [summary.assign(method=m) for m, summary in summaries.items()],
        ignore_index=True,
    )
    return law_table, fits


def run_two_room(N, steps, seeds, n, scenario, delta, bridges):
    q = make_q(scenario, n)
    rows = []
    for bridge in bridges:
        adj = two_room_graph(N, bridge=int(bridge), seed=0)
        res = simulate_two_room_source(
            adj, q, steps=steps, seeds=seeds, delta=delta, n_types=n
        )
        target = res["target_curves"]
        for m in THRESHOLD_METHODS:
            th50 = np.array([threshold_time(c, 0.5, 5, m) for c in target])
            th90 = np.array([threshold_time(c, 0.9, 5, m) for c in target])
            t50, h50 = th50[:, 0], th50[:, 1].astype(bool)
            t90, h90 = th90[:, 0], th90[:, 1].astype(bool)
            rows.append(
                {
                    "bridge": int(bridge),
                    "method": m,
                    "t50_median": conditional_median(t50, h50),
                    "hit50_rate": float(h50.mean()),
                    "rmst50": rmst(t50, h50, steps),
                    "t90_median": conditional_median(t90, h90),
                    "hit90_rate": float(h90.mean()),
                    "rmst90": rmst(t90, h90, steps),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    args = parser.parse_args()

    mode = "quick" if args.quick else "full"
    suffix = "_quick" if args.quick else ""
    graph = RUN_ALL[mode]["graph"]
    N, steps, seeds = graph["N"], graph["steps"], graph["seeds"]
    ws_p = graph.get("ws_p", WS_P)
    rgg_c = graph.get("rgg_c", RGG_C)
    n, scenario, delta = 10, "middle", 0.1

    if args.quick:
        tr = {"N": 120, "steps": 300, "seeds": 10, "bridges": [1, 4, 8]}
    else:
        tr = TWO_ROOM_SOURCE

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    started = time.time()

    law_table, fits = run_law(N, steps, seeds, n, scenario, delta, ws_p, rgg_c)
    law_table.to_csv(RESULTS / f"threshold_sensitivity_law{suffix}.csv", index=False)
    two_room = run_two_room(
        tr["N"], tr["steps"], tr["seeds"], n, scenario, delta, tr["bridges"]
    )
    two_room.to_csv(
        RESULTS / f"threshold_sensitivity_two_room{suffix}.csv", index=False
    )

    meta = {
        "mode": mode,
        "law": {"N": N, "steps": steps, "seeds": seeds, "n": n,
                "scenario": scenario, "delta": delta},
        "two_room": tr,
        "fits": fits,
    }
    (RESULTS / f"threshold_sensitivity_fit{suffix}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    summaries = {m: law_table[law_table["method"] == m] for m in THRESHOLD_METHODS}
    paths = plot_threshold_sensitivity(summaries, fits, two_room, FIGURES, suffix)
    write_metadata(
        RESULTS, mode,
        {"law": {**graph, "ws_p": ws_p, "rgg_c": rgg_c, "ws_k": 6}, "two_room": tr, "n": n, "scenario": scenario, "delta": delta, "methods": list(THRESHOLD_METHODS), "window": 5},
        suffix=suffix, name="threshold_sensitivity_metadata",
        extra={"graph_seed": 0, "swarm_rng": "default_rng(seed), seed=0..seeds-1; reused across configurations and definitions"},
        artifacts=[RESULTS / f"threshold_sensitivity_law{suffix}.csv", RESULTS / f"threshold_sensitivity_two_room{suffix}.csv", RESULTS / f"threshold_sensitivity_fit{suffix}.json", *paths],
    )

    print(f"[threshold sensitivity/{mode}] {time.time() - started:.1f}s")
    print("Law fits (t90 = intercept + slope * L):")
    for m in THRESHOLD_METHODS:
        f = fits[m]
        print(
            f"  {m:10s} intercept {f['intercept']:7.2f}  slope {f['slope']:6.2f}"
            f"  R2 {f['r2']:5.3f}  n {f['n_train']}"
        )
    print("Two-room RMST50 (censoring-aware):")
    pivot = two_room.pivot(index="bridge", columns="method", values="rmst50")
    print(pivot.to_string(float_format=lambda x: f"{x:8.2f}"))
    print("  wrote", ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()