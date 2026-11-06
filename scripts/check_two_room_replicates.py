"""Replicated source-room experiment: several graphs, several room sizes.

Splits the delay of the best type into the wait at the bridge (``t_first``)
and the spread inside the target room, and tests whether the wait scales with
the bridge conductance ``phi = bridge / (N / 2)``.

Usage:
    python scripts/check_two_room_replicates.py            # ~20 min
    python scripts/check_two_room_replicates.py --quick    # seconds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.bridge import (
    bootstrap_exponent,
    fit_loglog,
    fit_loglog2,
    run_replicates,
    summarise,
)
from hive.config import TWO_ROOM_REPLICATES
from hive.plots import plot_two_room_replicates

RESULTS = ROOT / "results"


def compute_fits(summary: pd.DataFrame, df: pd.DataFrame | None = None) -> dict[str, dict]:
    fits: dict[str, dict] = {}
    ok = summary[np.isfinite(summary["t_first_median"])]
    if len(ok) >= 3:
        # t_first ~ phi^b: a slope near -1 means the wait is set by conductance.
        fits["t_first_vs_phi"] = fit_loglog(ok["phi"], ok["t_first_median"])
        fits["t_first_vs_bridge_only"] = fit_loglog(ok["bridge"], ok["t_first_median"])
        fits["t_first_bridge_and_N"] = fit_loglog2(ok["bridge"], ok["N"], ok["t_first_median"])
        scaled = ok["t_first_median"] * ok["phi"]
        fits["t_first_times_phi"] = {
            "mean": float(scaled.mean()),
            "cv": float(scaled.std(ddof=0) / scaled.mean()),
        }
    ok50 = summary[np.isfinite(summary["t50_median"])]
    if len(ok50) >= 3:
        fits["t50_vs_phi"] = fit_loglog(ok50["phi"], ok50["t50_median"])
        fits["t50_bridge_and_N"] = fit_loglog2(ok50["bridge"], ok50["N"], ok50["t50_median"])
        post = ok50["t50_median"] - ok["t_first_median"].reindex(ok50.index)
        fits["share_of_t50_after_first_arrival"] = {
            "min": float((post / ok50["t50_median"]).min()),
            "max": float((post / ok50["t50_median"]).max()),
        }
    if df is not None:
        for name, key, hit in (
            ("t_first_vs_phi", "t_first", "hit_first"),
            ("t50_vs_phi", "t50", "hit50"),
        ):
            if name in fits:
                lo, hi = bootstrap_exponent(df, key, hit)
                fits[name]["slope_ci"] = [lo, hi]
    fits["path_length_range"] = {
        "min": float(summary["path_length"].min()),
        "max": float(summary["path_length"].max()),
    }
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    parser.add_argument("--refit", action="store_true", help="recompute fits from saved CSVs")
    parser.add_argument("--graph-seeds", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    suffix = "_quick" if args.quick else ""
    RESULTS.mkdir(exist_ok=True)
    started = time.time()

    if args.refit:
        df = pd.read_csv(RESULTS / f"two_room_replicates{suffix}.csv")
        summary = pd.read_csv(RESULTS / f"two_room_replicates_summary{suffix}.csv")
    else:
        cfg = TWO_ROOM_REPLICATES["quick" if args.quick else "full"]
        sizes = {int(k): int(v) for k, v in cfg["sizes"].items()}
        df = run_replicates(
            sizes,
            list(cfg["bridges"]),
            graph_seeds=args.graph_seeds or cfg["graph_seeds"],
            seeds=args.seeds or cfg["seeds"],
            progress=lambda msg: print(f"  [{time.time() - started:6.0f}s] {msg}", flush=True),
        )
        summary = summarise(df)
        df.to_csv(RESULTS / f"two_room_replicates{suffix}.csv", index=False)
        summary.to_csv(RESULTS / f"two_room_replicates_summary{suffix}.csv", index=False)

    fits = compute_fits(summary, df)
    (RESULTS / f"two_room_replicates_fits{suffix}.json").write_text(
        json.dumps(fits, indent=2), encoding="utf-8"
    )

    if not args.quick:
        figs = plot_two_room_replicates(summary, ROOT / "figures")
        print("  wrote", ", ".join(p.name for p in figs))

    cols = [
        "N",
        "bridge",
        "phi",
        "t_first_median",
        "t_first_hit_rate",
        "t50_median",
        "t50_hit_rate",
        "t90_median",
        "t90_hit_rate",
        "arrivals_mean",
        "extinctions_mean",
    ]
    print(f"[two-room replicates] {time.time() - started:.1f}s, {len(df)} runs")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
