"""Run both experiments end to end and write results/ and figures/.

Usage:
    python scripts/run_all.py            # full run (N=1000 / N=500, 100 seeds)
    python scripts/run_all.py --quick    # fast run for smoke testing
    python scripts/run_all.py --only graph
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hive.bandit import make_q
from hive.graph_sweep import WS_P, reference_mrd, run_graph_sweep
from hive.plots import config_summary, plot_law, plot_reproduction, plot_trajectories
from hive.reproduce import run_reproduction

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _reproduction(quick: bool) -> None:
    kwargs = (
        dict(N=300, steps=120, seeds=20)
        if quick
        else dict(N=1000, steps=300, seeds=100)
    )
    started = time.time()
    curves = run_reproduction(**kwargs)
    np.savez(RESULTS / "reproduction.npz", **curves)
    paths = plot_reproduction(curves, FIGURES)
    final = {k: float(v[:, -1].mean()) for k, v in curves.items()}
    print(
        f"[reproduction] {time.time() - started:.1f}s "
        f"final best mass: well-mixed={final['well_mixed']:.3f} "
        f"P-MCL={final['pmcl']:.3f} MRD={final['mrd']:.3f}"
    )
    print("  wrote", ", ".join(p.name for p in paths))


def _graph(quick: bool, replot: bool = False) -> None:
    kwargs = (
        dict(N=200, steps=120, seeds=12, ws_p=[0.01, 0.1, 1.0], rgg_c=[1.4, 2.0])
        if quick
        else dict(N=500, steps=400, seeds=100, ws_p=WS_P)
    )
    started = time.time()
    if replot:
        df = pd.read_csv(RESULTS / "graph_sweep.csv")
        stored = np.load(RESULTS / "graph_curves.npz")
        mean_curves = {k: stored[k] for k in stored.files}
    else:
        df, mean_curves = run_graph_sweep(**kwargs)
        df.to_csv(RESULTS / "graph_sweep.csv", index=False)
        np.savez(
            RESULTS / "graph_curves.npz",
            **{k.replace(":", "_"): v for k, v in mean_curves.items()},
        )
        print(f"[graph] {time.time() - started:.1f}s over {df.groupby(['family', 'param']).ngroups} configurations")

    summary = config_summary(df).sort_values(["family", "param"])
    summary.to_csv(RESULTS / "config_summary.csv", index=False)

    paths = plot_law(df, FIGURES)
    steps = len(next(iter(mean_curves.values()))) - 1
    paths += plot_trajectories(
        mean_curves, reference_mrd(make_q("middle"), steps), FIGURES
    )

    print(f"[graph] {time.time() - started:.1f}s over {len(summary)} configurations")
    print(
        summary[["family", "param", "tau", "t90", "hit_rate", "final"]]
        .to_string(index=False, float_format=lambda x: f"{x:8.2f}")
    )
    print("  wrote", ", ".join(p.name for p in paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small N and few seeds")
    parser.add_argument(
        "--only", choices=["reproduction", "graph"], default=None
    )
    parser.add_argument(
        "--replot", action="store_true", help="rebuild figures from stored results"
    )
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    if args.only in (None, "reproduction"):
        _reproduction(args.quick)
    if args.only in (None, "graph"):
        _graph(args.quick, replot=args.replot)


if __name__ == "__main__":
    main()
