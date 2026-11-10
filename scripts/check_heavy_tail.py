"""Supplementary check: does the t90 ~ L law survive heavy-tailed networks?

Three networks with N = 500, 100 seeds, 400 steps, and the same bandit as
Experiment B (scenario middle, n = 10 types):

- Barabasi-Albert, m = 2  (heavy tail)
- Barabasi-Albert, m = 5  (milder tail, denser)
- Erdos-Renyi, mean degree ~ 6 (Poisson tail, sanity anchor), conditioned on
  connectivity so that tau and L mean the same thing as in the main sweep

Writes results/heavy_tail.csv. The calibration line from the main sweep is
t90 = 44.85 + 3.46 * L, fitted on the 16 non-intervention configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
from scipy import sparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.bandit import make_q
from hive.config import HEAVY_TAIL
from hive.graphs import average_path_length, degree_stats, relaxation_time
from hive.graph_sweep import simulate_on_graph

N = HEAVY_TAIL["n"]
STEPS = HEAVY_TAIL["steps"]
SEEDS = HEAVY_TAIL["seeds"]
N_TYPES = HEAVY_TAIL["n_types"]
SCENARIO = HEAVY_TAIL["scenario"]
INTERCEPT = HEAVY_TAIL["intercept"]
SLOPE = HEAVY_TAIL["slope"]


def to_csr(graph: nx.Graph) -> sparse.csr_matrix:
    return nx.to_scipy_sparse_array(
        graph,
        nodelist=list(range(graph.number_of_nodes())),
        dtype=float,
        format="csr",
    )


def connected_er(n: int, m: int, seed: int = 0) -> nx.Graph:
    """ER draw conditioned on connectivity.

    An unconditioned G(n, m) at mean degree 6 is often fragmented; comparing
    its reachable-pairs-only path length against a calibration fitted on
    connected graphs would change the meaning of the test.
    """
    for attempt in range(seed, seed + 1000):
        graph = nx.gnm_random_graph(n, m, seed=attempt)
        if nx.is_connected(graph):
            return graph
    raise RuntimeError(f"no connected ER draw in 1000 attempts (n={n}, m={m})")


def main() -> int:
    q = make_q(SCENARIO, N_TYPES)
    builders = [
        ("ba_m2", lambda: nx.barabasi_albert_graph(N, 2, seed=0)),
        ("ba_m5", lambda: nx.barabasi_albert_graph(N, 5, seed=0)),
        ("er_k6", lambda: connected_er(N, N * 6 // 2, seed=0)),
    ]

    rows = []
    for name, build in builders:
        adj = to_csr(build())
        _, tau = relaxation_time(adj)
        path_length = average_path_length(adj)
        stats = degree_stats(adj)
        result = simulate_on_graph(
            adj, q, steps=STEPS, seeds=SEEDS, n_types=N_TYPES
        )
        median_t90 = float(np.median(result["t90"]))
        prediction = HEAVY_TAIL["intercept"] + HEAVY_TAIL["slope"] * path_length
        rows.append(
            {
                "family": name,
                "n": N,
                "tau": tau,
                "path_length": path_length,
                "degree_cv": stats["cv"],
                "t90": median_t90,
                "t90_mean": float(np.mean(result["t90"])),
                "hit_rate": float(np.mean(result["hit"])),
                "final": float(np.median(result["final"])),
                "final95": float(np.mean(result["final"] >= 0.95)),
                "t90_pred": prediction,
                "residual": median_t90 - prediction,
            }
        )
        print(
            "%-6s L=%6.3f tau=%9.2f t90=%6.1f pred=%6.1f "
            "resid=%+6.1f hit=%.2f f95=%.2f"
            % (
                name,
                path_length,
                tau,
                median_t90,
                prediction,
                median_t90 - prediction,
                rows[-1]["hit_rate"],
                rows[-1]["final95"],
            )
        )
        sys.stdout.flush()

    out = ROOT / "results" / "heavy_tail.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
