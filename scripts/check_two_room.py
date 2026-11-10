"""Source-room intervention on the two-room graph.

The original two-room sweep seeded every room with all ten types, so both
rooms already contained the best option and the bridge never carried useful
information. This script seeds the best option in one room only and measures
how long it takes to reach 10%, 50% and 90% of the target room.

Usage:
    python scripts/check_two_room.py
    python scripts/check_two_room.py --N 300 --steps 600 --seeds 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.bandit import make_q
from hive.config import TWO_ROOM_SOURCE
from hive.graph_sweep import simulate_two_room_source
from hive.graphs import average_path_length, degree_stats, relaxation_time, two_room_graph
from hive.metrics import rmst
from hive.plots import plot_two_room_source

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def median_ci(values: np.ndarray, n_boot: int = 800, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    median = float(np.median(values))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    boots = np.median(draws, axis=1)
    return median, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=TWO_ROOM_SOURCE["N"])
    parser.add_argument("--steps", type=int, default=TWO_ROOM_SOURCE["steps"])
    parser.add_argument("--seeds", type=int, default=TWO_ROOM_SOURCE["seeds"])
    parser.add_argument("--n", type=int, default=TWO_ROOM_SOURCE["n_types"])
    parser.add_argument("--scenario", default=TWO_ROOM_SOURCE["scenario"])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument(
        "--bridges",
        type=int,
        nargs="+",
        default=TWO_ROOM_SOURCE["bridges"],
    )
    parser.add_argument("--source-room", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    q = make_q(args.scenario, args.n)
    best = int(np.argmax(q))

    rows: list[dict[str, float | int | bool]] = []
    curves: dict[int, np.ndarray] = {}
    started = time.time()

    for bridge in args.bridges:
        adj = two_room_graph(args.N, bridge=int(bridge), seed=0)
        lambda2, tau = relaxation_time(adj)
        stats = degree_stats(adj)
        path_length = average_path_length(adj)
        res = simulate_two_room_source(
            adj,
            q,
            steps=args.steps,
            seeds=args.seeds,
            delta=args.delta,
            n_types=args.n,
            source_room=args.source_room,
        )
        curves[bridge] = res["target_curves"].mean(axis=0)
        for seed in range(args.seeds):
            rows.append(
                {
                    "bridge": bridge,
                    "seed": seed,
                    "N": args.N,
                    "tau": tau,
                    "lambda2": lambda2,
                    "path_length": path_length,
                    "degree_cv": stats["cv"],
                    "t10_target": res["t10_target"][seed],
                    "hit10_target": bool(res["hit10_target"][seed]),
                    "t50_target": res["t50_target"][seed],
                    "hit50_target": bool(res["hit50_target"][seed]),
                    "t90_target": res["t90_target"][seed],
                    "hit90_target": bool(res["hit90_target"][seed]),
                    "t90": res["t90"][seed],
                    "hit": bool(res["hit"][seed]),
                    "final_target": res["final_target"][seed],
                    "final": res["final"][seed],
                    "consensus": res["consensus"][seed],
                    "l2_mrd": res["l2_mrd"][seed],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "two_room_source.csv", index=False)

    summary_rows = []
    for bridge, group in df.groupby("bridge"):
        t50_values = group.loc[group["hit50_target"], "t50_target"]
        t90_values = group.loc[group["hit90_target"], "t90_target"]
        t50, t50_lo, t50_hi = median_ci(t50_values)
        t90_value, t90_lo, t90_hi = median_ci(t90_values)
        summary_rows.append(
            {
                "bridge": int(bridge),
                "tau": group["tau"].iloc[0],
                "path_length": group["path_length"].iloc[0],
                "t10_median": float(group["t10_target"].median()),
                "t50_median": t50,
                "t50_lo": t50_lo,
                "t50_hi": t50_hi,
                "t90_median": t90_value,
                "t90_lo": t90_lo,
                "t90_hi": t90_hi,
                "hit50_rate": float(group["hit50_target"].mean()),
                "hit90_rate": float(group["hit90_target"].mean()),
                "rmst50_target": rmst(
                    group["t50_target"], group["hit50_target"], args.steps
                ),
                "rmst90_target": rmst(
                    group["t90_target"], group["hit90_target"], args.steps
                ),
                "final_target_median": float(group["final_target"].median()),
                "final_target95": float((group["final_target"] >= 0.95).mean()),
                "t90_overall_median": float(group["t90"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("bridge")
    summary.to_csv(RESULTS / "two_room_source_summary.csv", index=False)

    paths = plot_two_room_source(df, curves, FIGURES)
    print(f"[two-room source] {time.time() - started:.1f}s")
    print(
        summary[
            [
                "bridge",
                "tau",
                "t50_median",
                "hit50_rate",
                "t90_median",
                "hit90_rate",
                "rmst50_target",
                "rmst90_target",
                "final_target_median",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:8.2f}")
    )
    print("  wrote", ", ".join(p.name for p in paths))
    print("  best arm:", best)


if __name__ == "__main__":
    main()
