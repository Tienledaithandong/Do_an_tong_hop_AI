"""Source-room experiment on rooms with uneven degrees.

Barabasi-Albert rooms have a few hubs. The bridge is attached to hubs, leaves
or random nodes of the source and of the target room, in all nine
combinations, plus a 6-regular control with the same mean degree. If the wait
for the best type to reach the target follows ``sum 1/deg(target endpoint)``
then target-side placement should matter a lot and source-side placement
little, and the room-average ratio ``bridge / mean degree`` cannot explain it.

Usage:
    python scripts/check_two_room_hetero.py            # ~8 min
    python scripts/check_two_room_hetero.py --quick    # seconds
    python scripts/check_two_room_hetero.py --refit    # reuse the saved CSV
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.config import TWO_ROOM_HETERO
from hive.hetero import (
    PREDICTORS,
    bootstrap_hetero,
    compare_hetero_predictors,
    placement_effects,
    run_hetero,
    summarise_hetero,
)
from hive.plots import plot_two_room_hetero
from hive.provenance import write_metadata

RESULTS = ROOT / "results"
PLACEMENTS = ("random", "hub", "leaf")


def build_layouts(m: int) -> list[dict]:
    layouts = [
        {"kind": "ba", "m": m, "source": s, "target": t}
        for s in PLACEMENTS
        for t in PLACEMENTS
    ]
    layouts.append({"kind": "regular", "m": m, "source": "random", "target": "random"})
    return layouts


def compute_fits(summary: pd.DataFrame, df: pd.DataFrame | None) -> dict:
    fits: dict = {}
    for key, hit in (("t_first", "hit_first"), ("t50", "hit50")):
        fit = compare_hetero_predictors(summary, f"{key}_median")
        if df is not None:
            fit["target_and_source"]["ci"] = bootstrap_hetero(
                df, key, hit, ["inv_end_target", "inv_end_source"]
            )
            fit["target_and_room"]["ci"] = bootstrap_hetero(
                df, key, hit, ["inv_end_target", "phi_deg_room"]
            )
            fit["target_and_tau"]["ci"] = bootstrap_hetero(
                df, key, hit, ["inv_end_target", "tau"]
            )
        fits[key] = fit
    fits["placement_effects_t_first"] = placement_effects(summary).to_dict("records")
    fits["placement_effects_t50"] = placement_effects(summary, "t50_median").to_dict("records")
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refit", action="store_true", help="recompute from saved CSV")
    parser.add_argument("--graph-seeds", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    suffix = "_quick" if args.quick else ""
    RESULTS.mkdir(exist_ok=True)
    started = time.time()
    path = RESULTS / f"two_room_hetero{suffix}.csv"

    if args.refit:
        df = pd.read_csv(path)
    else:
        cfg = TWO_ROOM_HETERO["quick" if args.quick else "full"]
        df = run_hetero(
            cfg["N"],
            cfg["steps"],
            list(cfg["bridges"]),
            build_layouts(cfg["m"]),
            graph_seeds=args.graph_seeds or cfg["graph_seeds"],
            seeds=args.seeds or cfg["seeds"],
            progress=lambda msg: print(f"  [{time.time() - started:6.0f}s] {msg}", flush=True),
        )
        df.to_csv(path, index=False)
        write_metadata(
            RESULTS, "quick" if args.quick else "full",
            {**cfg, "graph_seeds": args.graph_seeds or cfg["graph_seeds"], "seeds": args.seeds or cfg["seeds"], "layouts": build_layouts(cfg["m"]), "n_types": 10, "scenario": "middle", "delta": 0.1, "source_room": 0},
            suffix=suffix, name="two_room_hetero_metadata",
            extra={"graph_rng": "graph_seed=0..graph_seeds-1 per cell", "swarm_rng": "default_rng(seed), seed=0..seeds-1; reused across cells and graphs"},
            artifacts=[path],
        )

    summary = summarise_hetero(df)
    summary.to_csv(RESULTS / f"two_room_hetero_summary{suffix}.csv", index=False)
    fits = compute_fits(summary, df)
    (RESULTS / f"two_room_hetero_fits{suffix}.json").write_text(
        json.dumps(fits, indent=2), encoding="utf-8"
    )
    figs = []
    if not args.quick:
        figs = plot_two_room_hetero(summary, ROOT / "figures")
        print("  wrote", ", ".join(p.name for p in figs))
    write_metadata(
        RESULTS, "quick" if args.quick else "full",
        {"summary_n_boot": 800, "fit_n_boot": 300, "bootstrap_seed": 0},
        suffix=suffix, name="two_room_hetero_fits_metadata",
        inputs=[path],
        artifacts=[RESULTS / f"two_room_hetero_summary{suffix}.csv", RESULTS / f"two_room_hetero_fits{suffix}.json", *figs],
    )

    cols = [
        "layout",
        "bridge",
        "end_deg_target",
        "mean_deg_target",
        "inv_end_target",
        "inv_end_source",
        "tau",
        "t_first_median",
        "t_first_hit_rate",
        "t50_median",
        "t50_hit_rate",
    ]
    print(f"[two-room hetero] {time.time() - started:.1f}s, {len(df)} runs")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:9.3f}"))
    for key in ("t_first", "t50"):
        f = fits[key]
        print(f"\n{key}: single-predictor log-log fits ({f['cells']} cells)")
        for name in PREDICTORS:
            s = f["single"][name]
            print(f"  {name:15s} slope {s['slope']:+.3f}  R2 {s['r2']:.3f}")
        for name in ("target_and_source", "target_and_room", "target_and_tau"):
            m = f[name]
            parts = ", ".join(f"{k} {v:+.2f}" for k, v in m.items() if k.endswith("_exp"))
            print(f"  {name}: {parts}  R2 {m['r2']:.3f}  CI {m.get('ci')}")
    print("\nplacement effect on t_first (ratio to random->random):")
    print(
        pd.DataFrame(fits["placement_effects_t_first"]).to_string(
            index=False, float_format=lambda x: f"{x:6.2f}"
        )
    )


if __name__ == "__main__":
    main()
