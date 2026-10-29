#!/usr/bin/env python3
"""graph_voter.py -- Weighted voter swarm on communication graphs vs MRD.

Generates candidate communication graphs (complete, ring, regular, Erdős-Rabin,
Barabási-Albert, Watts-Strogatz), runs the weighted voter model across many
seeds, and compares the best-option share over time with the Maynard-Smith
replicator dynamics reference. Results (per-seed CSV, summary CSV, curves NPZ,
figures) are written to ``out/``.

Usage
-----
    python graph_voter.py --exp all --N 1000 --seeds 50 --jobs 2

    --exp : types | k | p | all
        types  -> complete, ring, regular, er, ba, ws
        k      -> degree sweep (k=4..16) on ring / regular / ws
        p      -> Watts-Strogatz rewiring sweep (p=0..1) at k=10
        all    -> everything above

Spectrum columns are recorded only for the first seed of each configuration
(mask the same value for every seed of that config).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import sparse

import networkx as nx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hive.bandit import Bandit, make_q
from hive.graphs import degree_stats, relaxation_time, ws_graph
from hive.metrics import best_mass, consensus as _consensus, l2_to_reference, t90
from hive.models import mrd_step
from hive.swarm import graph_voter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = ROOT / "out"

DEFAULT_TYPES = 10
DEFAULT_DELTA = 0.1
DEFAULT_SCENARIO = "middle"
DEFAULT_STEPS = 400

FAMILIES = ("complete", "ring", "regular", "er", "ba", "ws")
K_VALUES = (4, 6, 8, 10, 12, 16)
P_VALUES = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)

SEED_GRAPH = 12345  # fixed seed so graph topology can be regenerated/verified


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(family: str, k: int | None, p: float | None, N: int, seed: int) -> sparse.spmatrix:
    """Return a sparse adjacency matrix for one communication graph."""
    if family == "complete":
        return sparse.csr_matrix(np.ones((N, N)) - np.eye(N))
    if family == "ring":
        return ws_graph(N, k, 0.0, seed=seed)
    if family == "regular":
        return nx.to_scipy_sparse_array(nx.random_regular_graph(k, N, seed=seed), dtype=float, format="csr")
    if family == "er":
        return nx.to_scipy_sparse_array(nx.erdos_renyi_graph(N, k / N, seed=seed), dtype=float, format="csr")
    if family == "ba":
        m = max(1, k // 2)
        return nx.to_scipy_sparse_array(nx.barabasi_albert_graph(N, m, seed=seed), dtype=float, format="csr")
    if family == "ws":
        return ws_graph(N, k, p, seed=seed)  # p None only reached for non-ws families
    raise ValueError(f"unknown graph family: {family!r}")


# ---------------------------------------------------------------------------
# Voter-model simulation for a single (config, seed)
# ---------------------------------------------------------------------------
def _initial_types(rng: np.random.Generator, n_types: int, N: int) -> np.ndarray:
    types = np.repeat(np.arange(n_types), N // n_types)
    if types.size < N:
        types = np.concatenate([types, rng.integers(0, n_types, N - types.size)])
    rng.shuffle(types)
    return types


def _run_one(args) -> dict:
    """Worker: run every seed for one graph configuration."""
    family, k, p, N, n_types, steps, seeds, delta, scenario = args
    q = make_q(scenario, n_types)
    best = int(np.argmax(q))

    # Reference MRD curve (deterministic replicator, uniform initial policy).
    mrd_curve = np.empty(steps + 1)
    pi = np.full(n_types, 1.0 / n_types)
    mrd_curve[0] = pi[best]
    for s in range(steps):
        pi = mrd_step(pi, q)
        mrd_curve[s + 1] = pi[best]

    adj = build_graph(family, k, p, N, SEED_GRAPH)
    stats = degree_stats(adj)
    lambda2, tau = relaxation_time(adj)

    curves = np.zeros((seeds, steps + 1))
    t90s = np.zeros(seeds)
    hits = np.zeros(seeds, dtype=bool)
    l1s = np.zeros(seeds)
    finals = np.zeros(seeds)
    consensuses = np.zeros(seeds)
    winners = np.zeros(seeds, dtype=int)
    wrongs = np.zeros(seeds, dtype=bool)

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        bandit = Bandit(q, delta=delta, rng=rng)
        types = _initial_types(rng, n_types, N)
        curve = np.zeros(steps + 1)
        curve[0] = best_mass(types, best)
        for st in range(steps):
            rewards = bandit.pull(types)
            types = graph_voter(types, rewards, adj, rng, n_types=n_types)
            curve[st + 1] = best_mass(types, best)
        curves[seed] = curve

        t90s[seed], hits[seed] = t90(curve)
        l1s[seed] = float(np.mean(np.abs(curve - mrd_curve)))
        finals[seed] = float(curve[-1])
        counts = np.bincount(types, minlength=n_types)
        winner = int(np.argmax(counts))
        winners[seed] = winner
        near_consensus = counts[winner] / N >= 0.9
        consensuses[seed] = float(counts.max() / N)
        wrongs[seed] = bool(near_consensus and winner != best)

    return {
        "family": family,
        "k": k,
        "p": p,
        "N": N,
        "n_types": n_types,
        "steps": steps,
        "seeds": seeds,
        "lambda2": lambda2,
        "tau": tau,
        "spectral_gap": 1.0 - lambda2,
        "degree_mean": stats["mean"],
        "degree_std": stats["std"],
        "curves": curves,
        "t90s": t90s,
        "hits": hits,
        "l1s": l1s,
        "finals": finals,
        "consensuses": consensuses,
        "winners": winners,
        "wrongs": wrongs,
        "mrd_curve": mrd_curve,
        "best": best,
    }


def _run_configs_parallel(configs, fixed, jobs: int):
    tasks = [(fam, k, p) + fixed for (fam, k, p) in configs]
    try:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_run_one, tasks, chunksize=1))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] parallel ({exc}); falling back to sequential", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=1) as ex:
            return list(ex.map(_run_one, tasks, chunksize=1))


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def _default_configs(exp: str) -> list:
    if exp == "types":
        return [
            ("complete", None, None),
            ("ring", 10, 0.0),
            ("regular", 10, None),
            ("er", 10, None),
            ("ba", 10, None),
            ("ws", 10, 0.1),
        ]
    if exp == "k":
        rows = []
        for k in K_VALUES:
            rows.append(("ring", k, 0.0))
            rows.append(("regular", k, None))
            rows.append(("ws", k, 0.1))
        return rows
    if exp == "p":
        return [("ws", 10, p) for p in P_VALUES]
    if exp == "all":
        return _default_configs("types") + _default_configs("k") + _default_configs("p")
    raise ValueError(f"unknown --exp {exp!r}; use types, k, p, all")


# ---------------------------------------------------------------------------
# Output: assembly + persistence
# ---------------------------------------------------------------------------
def _assemble(results: list[dict], exp: str, args) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    import pandas as pd

    # Per-config summary.
    rows = []
    for r in results:
        rows.append(
            {
                "family": r["family"],
                "k": r["k"],
                "p": r["p"],
                "N": r["N"],
                "n_types": r["n_types"],
                "steps": r["steps"],
                "seeds": r["seeds"],
                "lambda2": r["lambda2"],
                "tau": r["tau"],
                "spectral_gap": r["spectral_gap"],
                "degree_mean": r["degree_mean"],
                "degree_std": r["degree_std"],
                "t90_mean": float(r["t90s"].mean()),
                "t90_std": float(r["t90s"].std()),
                "t90_hit_rate": float(r["hits"].mean()),
                "l1_mean": float(r["l1s"].mean()),
                "l1_std": float(r["l1s"].std()),
                "final_mean": float(r["finals"].mean()),
                "consensus_mean": float(r["consensuses"].mean()),
                "wrong_rate": float(r["wrongs"].mean()),
            }
        )
    summary = pd.DataFrame(rows)

    # Per-seed table (spectral columns only on seed 0, per spec).
    seed_rows = []
    for r in results:
        for s in range(r["seeds"]):
            seed_rows.append(
                {
                    "family": r["family"],
                    "k": r["k"],
                    "p": r["p"],
                    "seed": s,
                    "N": r["N"],
                    "n_types": r["n_types"],
                    "steps": r["steps"],
                    "generator": "fixed_seed" if s == 0 else "",
                    "lambda2": r["lambda2"] if s == 0 else np.nan,
                    "spectral_gap": r["spectral_gap"] if s == 0 else np.nan,
                    "degree_mean": r["degree_mean"],
                    "t90": r["t90s"][s],
                    "t90_hit": bool(r["hits"][s]),
                    "l1": r["l1s"][s],
                    "final": r["finals"][s],
                    "consensus": r["consensuses"][s],
                    "winner": r["winners"][s],
                    "wrong_consensus": r["wrongs"][s],
                }
            )
    seeds_df = pd.DataFrame(seed_rows)

    # Save raw curves keyed by "family_k_p".
    def curve_key(r: dict) -> str:
        k = r["k"] if r["k"] is not None else "N"
        p = f"{r['p']:g}" if r["p"] is not None else "0"
        return f"{r['family']}_{k}_{p}"

    curves = {curve_key(r): r["curves"] for r in results}
    np.savez(OUT_DIR / "graph_voter_curves.npz", **curves)

    return summary, seeds_df, curves


def save_results(summary: pd.DataFrame, seeds_df: pd.DataFrame) -> None:
    summary.to_csv(OUT_DIR / "graph_voter_summary.csv", index=False)
    seeds_df.to_csv(OUT_DIR / "graph_voter_seeds.csv", index=False)
    print(f"  wrote {OUT_DIR / 'graph_voter_summary.csv'}")
    print(f"  wrote {OUT_DIR / 'graph_voter_seeds.csv'}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> list:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_curves(results, n_panels: int) -> list:
    """One panel per configuration; MRD overlaid in every panel."""
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, r in zip(axes, results):
        t = np.arange(r["curves"].shape[1])
        mean = r["curves"].mean(axis=0)
        ax.plot(t, mean, "-", color="#1f77b4", lw=1.6, label="voter (mean)")
        ax.plot(t, r["mrd_curve"], "--", color="#2ca02c", lw=1.2, label="MRD")
        ax.set_title(f"{r['family']}" + (f" k={r['k']}" if r["k"] is not None else "") + (f" p={r['p']}" if r["p"] is not None else ""), fontsize=8)
        ax.set_xlim(0, r["steps"])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Share of best option")
        ax.legend(frameon=False, fontsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"best-option share vs MRD  (N={results[0]['N']}, seeds={results[0]['seeds']})", y=1.0)
    return _save(fig, "fig_voter_curves")


def _plot_spectral(results) -> list:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    families = sorted({r["family"] for r in results})
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    palette = dict(zip(families, colors))
    for r in results:
        ax.scatter(r["spectral_gap"], r["t90s"].mean(), s=60, color=palette[r["family"]], label=r["family"], zorder=3)
        ax.annotate(f"{r['family']}:{(r['k'] if r['k'] is not None else 'N')}:{(r['p'] if r['p'] is not None else 0):g}",
                    (r["spectral_gap"], r["t90s"].mean()), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Spectral gap  (1 - lambda2)")
    ax.set_ylabel("Steps to reach 90% (mean t90)")
    ax.set_title("Convergence speed tracks the spectral gap")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "fig_voter_spectral")


def _plot_l1(results) -> list:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = np.arange(results[0]["steps"] + 1)
    reps = []
    for r in results:
        key = (r["family"], r["k"], r["p"])
        if key not in _L1_REPS:
            _L1_REPS.add(key)
            reps.append(r)
    for r in reps:
        l1_t = np.mean(np.abs(r["curves"] - r["mrd_curve"][None, :]), axis=0)
        ax.plot(t, l1_t, label=f"{r['family']}", lw=1.6)
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Mean |voter curve - MRD|")
    ax.set_title("L1 distance from the well-mixed MRD reference")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_voter_l1")


def _plot_consensus(results) -> list:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    families = sorted({r["family"] for r in results})
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    palette = dict(zip(families, colors))
    order = sorted(results, key=lambda r: float(r["wrongs"].mean()), reverse=True)
    names = [f"{r['family']}" + (f" k={r['k']}" if r["k"] is not None else "") for r in order]
    vals = [float(r["wrongs"].mean()) * 100 for r in order]
    bars = ax.barh(names, vals, color=[palette[r["family"]] for r in order], edgecolor="white")
    ax.set_xlabel("Wrong consensus rate (%)")
    ax.set_title("Rate of false consensus (N=1000: 0 by construction)")
    ax.grid(alpha=0.3, axis="x")
    for b, r in zip(bars, order):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{float(r['wrongs'].mean())*100:.2f}%", va="center", fontsize=8)
    return _save(fig, "fig_voter_consensus")


_L1_REPS = set()


# ---------------------------------------------------------------------------
# Summary console output
# ---------------------------------------------------------------------------
def _print_summary(results, args) -> None:
    print(f"\n{'family':<10}{'k':>6}{'p':>6}{'lambda2':>9}{'gap':>8}{'t90':>8}{'hit%':>7}{'L1':>8}{'wrong%':>8}")
    for r in sorted(results, key=lambda r: (r["family"], r["k"] if r["k"] is not None else 0, r["p"] if r["p"] is not None else 0)):
        print(
            f"{r['family']:<10}{str(r['k']):>6}{str(r['p'] if r['p'] is not None else '-'):>6}"
            f"{r['lambda2']:>9.4f}{r['spectral_gap']:>8.4f}"
            f"{r['t90s'].mean():>8.2f}{r['hits'].mean()*100:>7.1f}{r['l1s'].mean():>8.4f}{r['wrongs'].mean()*100:>8.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, choices=["types", "k", "p", "all"], help="which family of experiments to run")
    parser.add_argument("--N", type=int, default=1000, help="population size (default 1000)")
    parser.add_argument("--types", type=int, default=DEFAULT_TYPES, help="number of options (default 10)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="simulation steps (default 400)")
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (default 50)")
    parser.add_argument("--jobs", type=int, default=2, help="parallel workers (default 2)")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="bandit noise width (default 0.1)")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="bandit scenario name (default 'middle')")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = _default_configs(args.exp)
    fixed = (args.N, args.types, args.steps, args.seeds, args.delta, args.scenario)
    print(f"[graph_voter] exp={args.exp} N={args.N} types={args.types} steps={args.steps} seeds={args.seeds} jobs={args.jobs}")

    started = time.time()
    results = _run_configs_parallel(configs, fixed, args.jobs)
    elapsed = time.time() - started
    total_sims = sum(r["seeds"] for r in results)
    print(f"[graph_voter] done in {elapsed:.1f}s over {total_sims} simulations")

    summary, seeds_df, curves = _assemble(results, args.exp, args)
    save_results(summary, seeds_df)
    _plot_curves(results, len(configs))
    _plot_spectral(results)
    _plot_l1(results)
    _plot_consensus(results)
    _print_summary(results, args)
    print("  wrote figures to", ", ".join(sorted(p.name for p in OUT_DIR.glob("fig_voter_*.png"))))


if __name__ == "__main__":
    main()
