"""Run both experiments end to end and write results/ and figures/.

Full runs write to ``results/`` and ``figures/``; ``--quick`` writes to
``results/quick/`` and ``figures/quick/`` so a smoke test can never overwrite
the full-run artifacts. Each experiment writes a metadata sidecar after a
simulation; ``--replot`` preserves the simulation provenance.

Usage:
    python scripts/run_all.py            # full run (N=1000 / N=500, 100 seeds)
    python scripts/run_all.py --quick    # fast run into results/quick/
    python scripts/run_all.py --only graph
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
from hive.config import BRIDGE_P, RUN_ALL
from hive.graph_sweep import reference_mrd, run_graph_sweep
from hive.plots import config_summary, plot_law, plot_reproduction, plot_trajectories
from hive.provenance import write_metadata
from hive.reproduce import run_reproduction

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def output_dirs(quick: bool) -> tuple[Path, Path]:
    """Quick artifacts live in a subdirectory so they cannot clobber full ones."""
    if quick:
        return RESULTS / "quick", FIGURES / "quick"
    return RESULTS, FIGURES


def _reproduction(quick: bool, replot: bool = False) -> None:
    kwargs = RUN_ALL["quick" if quick else "full"]["reproduction"]
    results, figures = output_dirs(quick)
    started = time.time()
    if replot:
        with np.load(results / "reproduction.npz") as stored:
            curves = {key: stored[key] for key in stored.files}
    else:
        curves = run_reproduction(**kwargs)
        np.savez(results / "reproduction.npz", **curves)
    paths = plot_reproduction(curves, figures)
    if not replot:
        write_metadata(
            results, "quick" if quick else "full",
            {"n": 10, "scenario": "middle", "delta": 0.1, "B": kwargs["N"], **kwargs},
            name="reproduction_metadata",
            extra={"swarm_rng": "default_rng(seed), seed=0..seeds-1; shared stream with P-MCL"},
            artifacts=[results / "reproduction.npz", *paths],
        )
    else:
        write_metadata(
            results, "quick" if quick else "full", {"operation": "replot"},
            name="reproduction_render_metadata", inputs=[results / "reproduction.npz"], artifacts=paths,
        )
    final = {k: float(v[:, -1].mean()) for k, v in curves.items()}
    print(
        f"[reproduction] {time.time() - started:.1f}s "
        f"final best mass: well-mixed={final['well_mixed']:.3f} "
        f"P-MCL={final['pmcl']:.3f} MRD={final['mrd']:.3f}"
    )
    print("  wrote", ", ".join(p.name for p in paths))


def _graph(quick: bool, replot: bool = False) -> None:
    kwargs = RUN_ALL["quick" if quick else "full"]["graph"]
    results, figures = output_dirs(quick)
    started = time.time()
    if replot:
        df = pd.read_csv(results / "graph_sweep.csv")
        with np.load(results / "graph_curves.npz") as stored:
            mean_curves = {k: stored[k] for k in stored.files}
    else:
        df, mean_curves = run_graph_sweep(**kwargs)
        df.to_csv(results / "graph_sweep.csv", index=False)
        np.savez(
            results / "graph_curves.npz",
            **{k.replace(":", "_"): v for k, v in mean_curves.items()},
        )
        print(f"[graph] {time.time() - started:.1f}s over {df.groupby(['family', 'param']).ngroups} configurations")

    summary = config_summary(df).sort_values(["family", "param"])
    if not replot:
        summary.to_csv(results / "config_summary.csv", index=False)

    paths = plot_law(df, figures)
    steps = len(next(iter(mean_curves.values()))) - 1
    paths += plot_trajectories(
        mean_curves, reference_mrd(make_q("middle"), steps), figures
    )

    if not replot:
        write_metadata(
            results, "quick" if quick else "full",
            {"n": 10, "scenario": "middle", "delta": 0.1, "ws_k": 6, "bridges": BRIDGE_P, **kwargs},
            name="graph_sweep_metadata",
            extra={"graph_seed": 0, "swarm_rng": "default_rng(seed), seed=0..seeds-1; reused across configurations"},
            artifacts=[results / "graph_sweep.csv", results / "graph_curves.npz", results / "config_summary.csv", *paths],
        )
    else:
        write_metadata(
            results, "quick" if quick else "full", {"operation": "replot"},
            name="graph_sweep_render_metadata",
            inputs=[results / "graph_sweep.csv", results / "graph_curves.npz"], artifacts=paths,
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

    results, figures = output_dirs(args.quick)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "reproduction"):
        _reproduction(args.quick, replot=args.replot)
    if args.only in (None, "graph"):
        _graph(args.quick, replot=args.replot)


if __name__ == "__main__":
    main()
