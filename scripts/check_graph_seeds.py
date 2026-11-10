"""Multi-graph-seed calibration of the ``t90 ~ L`` law.

The headline law was fit on a single graph per configuration (seed=0), so its
bootstrap CI only captured swarm-seed noise on that one graph. This sweep
redraws ``graph_seeds`` graphs per configuration and runs ``seeds`` swarm seeds
on each, then reports a hierarchical bootstrap CI that resamples graphs within
config and (t90, hit) pairs within graph. RGG draws are rejection-conditioned on
connectivity so every calibration point is a connected graph.

Swarm seeds are offset per (config, graph) so every trajectory is independent
(review #4). Deterministic topologies (complete, WS p=0) are run as a single
graph with ``graph_seeds * seeds`` seeds rather than counted as multiple
independent graphs, so the reported graph count is the number of *distinct*
topologies while the run count stays the total number of trajectories.

Usage:
    python scripts/check_graph_seeds.py
    python scripts/check_graph_seeds.py --quick
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

from hive.analysis import hierarchical_bootstrap_law
from hive.bandit import make_q
from hive.config import GRAPH_SEEDS
from hive.graph_sweep import simulate_on_graph
from hive.graphs import (
    average_path_length,
    degree_stats,
    is_connected,
    relaxation_time,
    rgg_graph,
    ws_graph,
)
from hive.metrics import rmst
from hive.plots import plot_graph_seeds

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def connected_rgg(n: int, radius: float, seed: int) -> tuple[sparse.csr_matrix, int]:
    for attempt in range(seed, seed + 1000):
        adj = rgg_graph(n, radius, seed=attempt)
        if is_connected(adj):
            return adj, attempt
    raise RuntimeError(f"no connected RGG draw in 1000 attempts (n={n}, r={radius:.4f})")


def build_configs(cfg: dict) -> list[tuple[str, float, str]]:
    """(family, param, mode) specs; the actual graph draw is seeded per graph."""
    configs: list[tuple[str, float, str]] = [("complete", 1.0, "well_mixed")]
    for p in cfg["ws_p"]:
        configs.append(("ws", float(p), "graph"))
    for c in cfg["rgg_c"]:
        configs.append(("rgg", float(c), "graph"))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()
    cfg = GRAPH_SEEDS["quick" if args.quick else "full"]
    suffix = "_quick" if args.quick else ""

    N, steps = cfg["N"], cfg["steps"]
    G, S = cfg["graph_seeds"], cfg["seeds"]
    n, scenario = cfg["n_types"], cfg["scenario"]
    q = make_q(scenario, n)
    rc = float(np.sqrt(np.log(N) / (np.pi * N)))
    dens = sparse.csr_matrix(np.ones((N, N)) - np.eye(N))

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    started = time.time()

    rows: list[dict] = []
    graph_rows: list[dict] = []
    t90_bank: dict[str, np.ndarray] = {}
    specs = build_configs(cfg)
    seed_offset = 0
    for family, param, mode in specs:
        config = f"{family}:{param:g}"
        # Complete and WS p=0 are deterministic: redrawing them yields the same
        # graph, so G replicas are NOT G independent graphs. Run one draw with
        # G*S independent seeds instead and count it as a single graph.
        deterministic = family == "complete" or (family == "ws" and param == 0.0)
        G_eff = 1 if deterministic else G
        S_eff = G * S if deterministic else S
        for graph_seed in range(G_eff):
            if family == "complete":
                adj, draw_seed = dens, 0
            elif family == "ws":
                adj, draw_seed = ws_graph(N, 6, param, seed=graph_seed), graph_seed
            else:
                adj, draw_seed = connected_rgg(N, param * rc, seed=graph_seed * 1000)
            lambda2, tau = relaxation_time(adj)
            stats = degree_stats(adj)
            length = average_path_length(adj)
            res = simulate_on_graph(
                adj, q, steps=steps, seeds=S_eff, delta=0.1, n_types=n, mode=mode,
                seed_offset=seed_offset,
            )
            t90s = res["t90"]
            hits = res["hit"]
            t90_bank[f"{config}:{graph_seed}"] = t90s
            hit_rate = float(hits.mean())
            reached = t90s[hits]
            graph_rows.append(
                {
                    "config": config,
                    "family": family,
                    "param": param,
                    "graph_seed": graph_seed,
                    "draw_seed": draw_seed,
                    "deterministic": deterministic,
                    "N": N,
                    "n_seeds": S_eff,
                    "path_length": length,
                    "tau": tau,
                    "lambda2": lambda2,
                    "degree_mean": stats["mean"],
                    "degree_cv": stats["cv"],
                    "t90_median": float(np.median(reached)) if reached.size else float("nan"),
                    "hit_rate": hit_rate,
                    "rmst": rmst(t90s, hits, steps),
                }
            )
            for i in range(S_eff):
                rows.append(
                    {
                        "config": config,
                        "family": family,
                        "param": param,
                        "graph_seed": graph_seed,
                        "seed": seed_offset + i,
                        "path_length": length,
                        "tau": tau,
                        "t90": float(t90s[i]),
                        "t90_hit": bool(hits[i]),
                    }
                )
            seed_offset += S_eff
        print(f"  {config:16s} done ({time.time() - started:.0f}s)", flush=True)

    seeds_df = pd.DataFrame(rows)
    graphs_df = pd.DataFrame(graph_rows)
    seeds_df.to_csv(RESULTS / f"graph_seeds{suffix}.csv", index=False)
    graphs_df.to_csv(RESULTS / f"graph_seeds_summary{suffix}.csv", index=False)
    np.savez_compressed(RESULTS / f"graph_seeds_t90{suffix}.npz", **t90_bank)

    # Mirror law_frame: a graph whose seeds mostly censor carries no usable
    # median, so drop it before fitting (and report how many were dropped).
    usable = graphs_df[graphs_df["hit_rate"] >= 0.5]
    dropped = len(graphs_df) - len(usable)
    keep = seeds_df.merge(
        usable[["config", "graph_seed"]], on=["config", "graph_seed"], how="inner"
    )
    law = hierarchical_bootstrap_law(keep, n_boot=args.n_boot, seed=0)
    law["graphs_dropped_low_hit"] = int(dropped)
    with open(RESULTS / f"graph_seeds_law{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(law, fh, ensure_ascii=False, indent=2)

    paths = plot_graph_seeds(graphs_df, law, FIGURES, suffix=suffix)
    n_det = int(graphs_df["deterministic"].sum())
    print(
        f"[graph-seeds] {time.time() - started:.1f}s, "
        f"{len(graphs_df)} distinct graphs ({n_det} deterministic), "
        f"{len(seeds_df)} independent runs"
    )
    if "slope" in law:
        ci = law["ci"]["slope"]
        print(
            f"  law: t90 = {law['intercept']:.2f} + {law['slope']:.2f} * L "
            f"(R2={law['r2']:.3f}); slope 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
        )
    print("  wrote", ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()