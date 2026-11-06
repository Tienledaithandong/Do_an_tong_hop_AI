"""Separate tau from the bridge-per-node ratio with sparse rooms.

With clique rooms, bridge width sets tau and the bridge-per-node ratio
``phi`` together, so the two cannot be told apart. Replacing the cliques by
random regular graphs lowers tau by roughly ``degree / (N / 2)`` at the same
bridge width and the same ``phi``. This script runs that grid, pools it with
the clique grid from ``check_two_room_replicates.py`` and asks which predictor
explains the wait at the bridge: ``phi``, ``tau``, the cut conductance
``phi_vol`` or the per-neighbour ratio ``phi_deg = bridge / degree``.

Usage:
    python scripts/check_two_room_sparse.py            # ~25 min
    python scripts/check_two_room_sparse.py --quick    # seconds
    python scripts/check_two_room_sparse.py --refit    # reuse saved CSVs
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
    bootstrap_multi,
    compare_predictors,
    fit_loglog_multi,
    run_replicates,
    summarise,
    with_room_columns,
)
from hive.config import TWO_ROOM_SPARSE
from hive.plots import plot_two_room_sparse

RESULTS = ROOT / "results"
PREDICTORS = ["phi", "phi_vol", "phi_deg", "tau"]


def _ratio(a: float, b: float) -> float:
    """``a / b`` that is NaN unless both medians are observed (finite)."""
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else float("nan")


def compute_fits(summary: pd.DataFrame, df: pd.DataFrame | None) -> dict:
    fits: dict = {}
    for key, hit in (("t_first_median", "hit_first"), ("t50_median", "hit50")):
        ok = summary[np.isfinite(summary[key])]
        entry: dict = {"single": compare_predictors(summary, key, PREDICTORS)}
        if len(ok) > 4:
            entry["phi_and_tau"] = fit_loglog_multi(
                {"phi": ok["phi"], "tau": ok["tau"]}, ok[key]
            )
            entry["phi_deg_and_tau"] = fit_loglog_multi(
                {"phi_deg": ok["phi_deg"], "tau": ok["tau"]}, ok[key]
            )
            entry["phi_deg_and_phi"] = fit_loglog_multi(
                {"phi_deg": ok["phi_deg"], "phi": ok["phi"]}, ok[key]
            )
        if df is not None:
            raw = key.removesuffix("_median")
            for name, preds in (
                ("phi_and_tau", ["phi", "tau"]),
                ("phi_deg_and_tau", ["phi_deg", "tau"]),
                ("phi_deg_and_phi", ["phi_deg", "phi"]),
            ):
                if name in entry:
                    cis = bootstrap_multi(df, raw, hit, preds)
                    entry[name]["ci"] = {p: list(v) for p, v in cis.items()}
            entry["phi_deg_slope_ci"] = list(bootstrap_exponent(df, raw, hit, predictor="phi_deg"))
        fits[key.removesuffix("_median")] = entry

    # Paired contrast: same (N, bridge) so phi is identical; only the rooms differ.
    base = summary[summary["room_degree"] == 0].set_index(["N", "bridge"])
    pairs = []
    for degree, sub in summary[summary["room_degree"] > 0].groupby("room_degree"):
        sub = sub.set_index(["N", "bridge"])
        joined = sub.join(base, rsuffix="_clique", how="inner")
        for (N, bridge), row in joined.iterrows():
            pairs.append(
                {
                    "N": int(N),
                    "bridge": int(bridge),
                    "room_degree": int(degree),
                    "tau_ratio": _ratio(row["tau"], row["tau_clique"]),
                    "t_first_ratio": _ratio(row["t_first_median"], row["t_first_median_clique"]),
                    "t50_ratio": _ratio(row["t50_median"], row["t50_median_clique"]),
                    "degree_ratio": _ratio(row["degree"], row["degree_clique"]),
                }
            )
    fits["paired_vs_clique"] = pairs
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refit", action="store_true", help="recompute fits from saved CSVs")
    parser.add_argument("--graph-seeds", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    suffix = "_quick" if args.quick else ""
    RESULTS.mkdir(exist_ok=True)
    started = time.time()
    sparse_path = RESULTS / f"two_room_sparse{suffix}.csv"
    clique_path = RESULTS / f"two_room_replicates{suffix}.csv"

    if args.refit:
        sparse_df = pd.read_csv(sparse_path)
    else:
        cfg = TWO_ROOM_SPARSE["quick" if args.quick else "full"]
        sizes = {int(k): int(v) for k, v in cfg["sizes"].items()}
        frames = []
        for degree in cfg["room_degrees"]:
            frames.append(
                run_replicates(
                    sizes,
                    list(cfg["bridges"]),
                    graph_seeds=args.graph_seeds or cfg["graph_seeds"],
                    seeds=args.seeds or cfg["seeds"],
                    room_degree=int(degree),
                    progress=lambda msg: print(f"  [{time.time() - started:6.0f}s] {msg}", flush=True),
                )
            )
        sparse_df = pd.concat(frames, ignore_index=True)
        sparse_df.to_csv(sparse_path, index=False)

    if not clique_path.exists():
        raise SystemExit(
            f"{clique_path.name} not found; run scripts/check_two_room_replicates.py first"
        )
    pooled = pd.concat(
        [with_room_columns(pd.read_csv(clique_path)), with_room_columns(sparse_df)],
        ignore_index=True,
    )
    summary = summarise(pooled)
    summary.to_csv(RESULTS / f"two_room_sparse_summary{suffix}.csv", index=False)
    fits = compute_fits(summary, pooled)
    (RESULTS / f"two_room_sparse_fits{suffix}.json").write_text(
        json.dumps(fits, indent=2), encoding="utf-8"
    )

    if not args.quick:
        figs = plot_two_room_sparse(summary, ROOT / "figures")
        print("  wrote", ", ".join(p.name for p in figs))

    cols = [
        "N",
        "room_degree",
        "bridge",
        "tau",
        "phi",
        "phi_deg",
        "t_first_median",
        "t50_median",
        "t50_hit_rate",
        "arrivals_mean",
        "extinctions_mean",
    ]
    print(f"[two-room sparse] {time.time() - started:.1f}s, {len(pooled)} runs")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:9.3f}"))
    for key in ("t_first", "t50"):
        print(f"\n{key}: single-predictor log-log fits")
        for name, f in fits[key]["single"].items():
            print(f"  {name:8s} slope {f['slope']:+.3f}  R2 {f['r2']:.3f}")
        for name in ("phi_and_tau", "phi_deg_and_tau", "phi_deg_and_phi"):
            if name in fits[key]:
                f = fits[key][name]
                parts = ", ".join(f"{k} {v:+.2f}" for k, v in f.items() if k.endswith("_exp"))
                print(f"  {name}: {parts}  R2 {f['r2']:.3f}  CI {f.get('ci')}")


if __name__ == "__main__":
    main()
