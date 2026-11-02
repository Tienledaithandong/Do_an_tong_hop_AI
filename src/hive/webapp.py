"""Local web dashboard for the HiveMindRL experiments.

The server uses only the Python standard library for HTTP so the project keeps
its small dependency set. Simulations run in a single background worker; the
browser polls a job id and fetches PNG plots when the run finishes.

Start it with ``python scripts/run_ui.py`` (or ``python -m hive.webapp`` when
``src`` is on ``sys.path``). The server binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import sparse

from .bandit import SCENARIOS, make_q
from .analysis import law_diagnostics
from .config import BRIDGE_P, HEAVY_TAIL, RGG_C, RUN_ALL, WS_P
from .graph_sweep import reference_mrd, simulate_on_graph
from .graphs import (
    average_path_length,
    degree_stats,
    is_connected,
    relaxation_time,
    two_room_graph,
)
from .metrics import t90
from .plots import COLORS, config_summary, fit_law
from .reproduce import run_reproduction

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parent / "ui"
RESULTS_DIR = ROOT / "results"
REPORT_PDF = ROOT / "report" / "report.pdf"

MAX_BODY = 1 << 20
JOB_HISTORY = 24

FAMILIES: list[dict[str, Any]] = [
    {
        "id": "complete",
        "label": "Đồ thị đầy đủ",
        "short": "Đầy đủ",
        "param_label": "—",
        "default": 1.0,
        "min": 1.0,
        "max": 1.0,
        "step": 1.0,
        "integer": False,
        "hint": "Mọi cá thể quan sát toàn bộ đàn; τ ≈ 1. Đây là mốc trộn đều.",
    },
    {
        "id": "ws",
        "label": "Watts–Strogatz",
        "short": "WS",
        "param_label": "p (xác suất nối lại)",
        "default": 0.1,
        "min": 0.0,
        "max": 1.0,
        "step": 0.001,
        "integer": False,
        "hint": "p = 0 là vòng lattice bậc 6; p = 1 là đồ thị ngẫu nhiên.",
    },
    {
        "id": "rgg",
        "label": "Random geometric",
        "short": "RGG",
        "param_label": "c (hệ số bán kính)",
        "default": 2.0,
        "min": 0.5,
        "max": 3.5,
        "step": 0.1,
        "integer": False,
        "hint": "Bán kính = c·√(log N / (πN)); c nhỏ dễ phân mảnh.",
    },
    {
        "id": "two_room",
        "label": "Hai buồng + cầu",
        "short": "Hai buồng",
        "param_label": "Số cạnh cầu",
        "default": 6,
        "min": 1,
        "max": 12,
        "step": 1,
        "integer": True,
        "hint": "Hai clique nối bằng cầu hẹp: τ rất lớn nhưng L ≈ 2.",
    },
    {
        "id": "ba",
        "label": "Barabási–Albert",
        "short": "BA",
        "param_label": "m (cạnh ưu tiên)",
        "default": 2,
        "min": 1,
        "max": 8,
        "step": 1,
        "integer": True,
        "hint": "Mạng đuôi nặng; dùng để kiểm tra ngoài phạm vi quét chính.",
    },
    {
        "id": "er",
        "label": "Erdős–Rényi",
        "short": "ER",
        "param_label": "Bậc trung bình",
        "default": 6,
        "min": 2,
        "max": 20,
        "step": 1,
        "integer": True,
        "hint": "Đồ thị ngẫu nhiên G(n, m); điểm neo đuôi Poisson.",
    },
]
FAMILY_BY_ID = {family["id"]: family for family in FAMILIES}

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "label": "Nhanh (vài giây)",
        "reproduction": {"N": 300, "steps": 120, "seeds": 10},
        "graph": {"N": 150, "steps": 120, "seeds": 8},
        "compare": {"N": 120, "steps": 120, "seeds": 6},
    },
    "medium": {
        "label": "Vừa (~30 giây)",
        "reproduction": {"N": 600, "steps": 200, "seeds": 30},
        "graph": {"N": 300, "steps": 200, "seeds": 20},
        "compare": {"N": 200, "steps": 200, "seeds": 12},
    },
    "full": {
        "label": "Đầy đủ (vài phút)",
        "reproduction": {"N": 1000, "steps": 300, "seeds": 100},
        "graph": {"N": 500, "steps": 400, "seeds": 100},
        "compare": {"N": 400, "steps": 300, "seeds": 40},
    },
}

DEFAULT_COMPARE = [
    {"family": "complete", "param": 1.0},
    {"family": "ws", "param": 0.0},
    {"family": "ws", "param": 0.01},
    {"family": "ws", "param": 0.1},
    {"family": "ws", "param": 1.0},
    {"family": "rgg", "param": 1.4},
    {"family": "rgg", "param": 2.0},
    {"family": "two_room", "param": 1.0},
    {"family": "two_room", "param": 6.0},
]

FAMILY_COLORS = {
    "complete": "#3a3f47",
    "ws": COLORS["ws"],
    "rgg": COLORS["rgg"],
    "two_room": COLORS["two_room"],
    "ba": "#9467bd",
    "er": "#8c564b",
}

STORED_RESULTS = (
    "config_summary",
    "heavy_tail",
    "graph_sweep",
    "sample_config",
    "two_room_source",
    "two_room_source_summary",
)


class ValidationError(ValueError):
    """Raised when a request payload is outside the supported range."""


# ---------------------------------------------------------------------------
# Input validation


def _number(
    payload: dict, key: str, default: float, lo: float, hi: float
) -> float:
    value = payload.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{key} phải là số") from exc
    if not math.isfinite(number) or number < lo or number > hi:
        raise ValidationError(f"{key} phải nằm trong [{lo:g}, {hi:g}]")
    return number


def _integer(payload: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = payload.get(key, default)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{key} phải là số nguyên") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValidationError(f"{key} phải là số nguyên")
    if number < lo or number > hi:
        raise ValidationError(f"{key} phải nằm trong [{lo}, {hi}]")
    return number


def _common(payload: dict) -> dict[str, Any]:
    scenario = str(payload.get("scenario", "middle"))
    if scenario not in SCENARIOS:
        raise ValidationError(
            "scenario phải là một trong " + ", ".join(sorted(SCENARIOS))
        )
    return {
        "N": _integer(payload, "N", 150, 20, 800),
        "steps": _integer(payload, "steps", 120, 10, 1000),
        "seeds": _integer(payload, "seeds", 8, 1, 200),
        "n": _integer(payload, "n", 10, 2, 30),
        "scenario": scenario,
        "delta": _number(payload, "delta", 0.1, 0.0, 0.5),
    }


def _family_param(family: str, param: Any, N: int) -> tuple[str, float]:
    meta = FAMILY_BY_ID.get(family)
    if meta is None:
        raise ValidationError(
            "family phải là một trong " + ", ".join(sorted(FAMILY_BY_ID))
        )
    if family == "complete":
        return family, 1.0

    try:
        value = float(param)
    except (TypeError, ValueError) as exc:
        raise ValidationError("param phải là số") from exc
    if not math.isfinite(value):
        raise ValidationError("param phải hữu hạn")

    if meta["integer"]:
        if not float(value).is_integer():
            raise ValidationError(f"param của {meta['label']} phải là số nguyên")
        value = int(value)

    lo = float(meta["min"])
    hi = float(meta["max"])
    if family == "two_room":
        hi = min(hi, N // 2)
    if family == "ba":
        hi = min(hi, N - 1)
    if family == "er":
        hi = min(hi, N - 1)
    if value < lo or value > hi:
        raise ValidationError(
            f"param của {meta['label']} phải nằm trong [{lo:g}, {hi:g}]"
        )
    return family, value


def _config_label(family: str, param: float) -> str:
    if family == "complete":
        return "Đồ thị đầy đủ"
    if family == "ws":
        return f"WS · p={param:g}"
    if family == "rgg":
        return f"RGG · c={param:g}"
    if family == "two_room":
        return f"Hai buồng · cầu={int(param)}"
    if family == "ba":
        return f"BA · m={int(param)}"
    return f"ER · ⟨k⟩={int(param)}"


# ---------------------------------------------------------------------------
# Graph construction and layouts


def _normalise_positions(pos: dict[int, Any]) -> dict[int, tuple[float, float]]:
    xs = np.array([float(p[0]) for p in pos.values()])
    ys = np.array([float(p[1]) for p in pos.values()])
    span_x = float(xs.max() - xs.min()) or 1.0
    span_y = float(ys.max() - ys.min()) or 1.0
    return {
        int(node): (
            0.05 + 0.9 * (float(p[0]) - float(xs.min())) / span_x,
            0.05 + 0.9 * (float(p[1]) - float(ys.min())) / span_y,
        )
        for node, p in pos.items()
    }


def _circular_positions(n: int) -> dict[int, tuple[float, float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return {
        i: (0.5 + 0.46 * float(np.cos(a)), 0.5 + 0.46 * float(np.sin(a)))
        for i, a in enumerate(angles)
    }


def _two_room_positions(n: int, bridge: int) -> dict[int, tuple[float, float]]:
    half = n // 2
    pos: dict[int, tuple[float, float]] = {}
    for start, count, cx in ((0, half, 0.24), (half, n - half, 0.76)):
        for j in range(count):
            angle = 2.0 * np.pi * j / max(count, 1)
            radius = 0.19 * (0.72 + 0.28 * ((j % 3) / 2.0))
            pos[start + j] = (
                cx + radius * float(np.cos(angle)),
                0.5 + radius * float(np.sin(angle)),
            )
    return pos


def _to_adj(graph: nx.Graph, n: int) -> sparse.csr_matrix:
    return nx.to_scipy_sparse_array(
        graph, nodelist=list(range(n)), dtype=float, format="csr"
    )


def build_graph(
    family: str, param: float, N: int, seed: int = 0
) -> tuple[sparse.csr_matrix, dict[int, tuple[float, float]]]:
    """Build a communication graph and a layout suitable for the preview plot."""
    if family == "complete":
        dense = np.ones((N, N)) - np.eye(N)
        return sparse.csr_matrix(dense), _circular_positions(N)

    if family == "ws":
        k = min(6, N - 1)
        if k % 2:
            k -= 1
        k = max(k, 2)
        graph = nx.watts_strogatz_graph(N, k, float(param), seed=seed)
        return _to_adj(graph, N), _circular_positions(N)

    if family == "rgg":
        rc = float(np.sqrt(np.log(N) / (np.pi * N)))
        graph = nx.random_geometric_graph(N, float(param) * rc, seed=seed)
        raw = nx.get_node_attributes(graph, "pos")
        pos = _normalise_positions(raw) if raw else _circular_positions(N)
        return _to_adj(graph, N), pos

    if family == "two_room":
        bridge = int(param)
        adj = two_room_graph(N, bridge=bridge, seed=seed)
        return adj, _two_room_positions(N, bridge)

    if family == "ba":
        graph = nx.barabasi_albert_graph(N, int(param), seed=seed)
    elif family == "er":
        edges = max(N - 1, N * int(param) // 2)
        graph = nx.gnm_random_graph(N, edges, seed=seed)
    else:  # pragma: no cover - guarded by validation
        raise ValidationError(f"Đồ thị không hỗ trợ: {family}")

    iterations = 60 if N <= 300 else 25
    raw = nx.spring_layout(graph, seed=seed, iterations=iterations)
    return _to_adj(graph, N), _normalise_positions(raw)


def graph_metrics(adj: sparse.csgraph.spmatrix) -> dict[str, Any]:
    lambda2, tau = relaxation_time(adj)
    stats = degree_stats(adj)
    coo = sparse.triu(sparse.csr_matrix(adj), k=1).tocoo()
    return {
        "nodes": int(adj.shape[0]),
        "edges": int(coo.nnz),
        "connected": bool(is_connected(adj)),
        "lambda2": lambda2,
        "tau": tau,
        "path_length": float(average_path_length(adj)),
        "degree_mean": float(stats["mean"]),
        "degree_cv": float(stats["cv"]),
        "degree_min": float(stats["min"]),
        "degree_max": float(stats["max"]),
    }


def _edges(adj: sparse.spmatrix) -> list[tuple[int, int]]:
    coo = sparse.triu(sparse.csr_matrix(adj), k=1).tocoo()
    return list(zip(coo.row.tolist(), coo.col.tolist()))


# ---------------------------------------------------------------------------
# Plots


def _png(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)


def _plot_reproduction(curves: dict[str, np.ndarray]) -> bytes:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    styles = {
        "well_mixed": ("Bầy đàn (weighted voter)", "-", COLORS["well_mixed"]),
        "pmcl": ("P-MCL", "-", COLORS["pmcl"]),
        "mrd": ("MRD (tham chiếu)", "--", COLORS["mrd"]),
    }
    steps = curves["well_mixed"].shape[1]
    t = np.arange(steps)
    for key, (label, style, color) in styles.items():
        mean = curves[key].mean(axis=0)
        std = curves[key].std(axis=0)
        ax.plot(t, mean, style, color=color, label=label, linewidth=1.9)
        if key != "mrd":
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Bước quyết định", fontsize=10)
    ax.set_ylabel("Tỉ lệ chọn phương án tốt nhất", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, steps - 1)
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    return _png(fig)


def _plot_graph_curves(mean: np.ndarray, std: np.ndarray, mrd: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    t = np.arange(mean.size)
    ax.plot(t, mean, color="#0e7c86", linewidth=2.0, label="Bầy đàn trên đồ thị")
    ax.fill_between(
        t, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
        color="#0e7c86", alpha=0.16, linewidth=0,
    )
    ax.plot(t[: mrd.size], mrd, "--", color=COLORS["mrd"], linewidth=1.5, label="MRD (trộn đều)")
    ax.set_xlabel("Bước quyết định", fontsize=10)
    ax.set_ylabel("Tỉ lệ chọn phương án tốt nhất", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    return _png(fig)


def _plot_graph(
    adj: sparse.spmatrix,
    pos: dict[int, tuple[float, float]],
    types: np.ndarray,
    best: int,
    n_types: int,
) -> bytes:
    n = adj.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    edge_list = _edges(adj)
    if len(edge_list) > 3500:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(edge_list), size=3500, replace=False)
        edge_list = [edge_list[i] for i in pick]
    segments = [
        [pos[int(u)], pos[int(v)]] for u, v in edge_list if u in pos and v in pos
    ]
    if segments:
        ax.add_collection(
            LineCollection(segments, colors="#9aa0a6", linewidths=0.35, alpha=0.16, zorder=1)
        )

    cmap = plt.get_cmap("tab20")
    colors = [cmap((int(t) * 2 + 1) % 20) for t in types]
    colors = ["#e6a700" if int(t) == best else c for t, c in zip(types, colors)]
    xs = np.array([pos[i][0] for i in range(n)])
    ys = np.array([pos[i][1] for i in range(n)])
    sizes = 34 if n <= 250 else 16
    ax.scatter(xs, ys, c=colors, s=sizes, edgecolors="white", linewidths=0.45, zorder=2)
    best_nodes = np.flatnonzero(types == best)
    if best_nodes.size:
        ax.scatter(
            xs[best_nodes], ys[best_nodes], s=sizes * 2.1, facecolors="none",
            edgecolors="#8a5b00", linewidths=0.9, zorder=3,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return _png(fig)


def _plot_compare(rows: list[dict[str, Any]], fit: dict[str, Any]) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    def scatter(ax: plt.Axes, x_key: str, logx: bool) -> None:
        for row in rows:
            if row[x_key] is None:
                continue
            color = FAMILY_COLORS.get(row["family"], "#555555")
            censored = row["hit_rate"] < 0.5
            ax.scatter(
                row[x_key],
                row["t90_median"],
                s=52,
                color="none" if censored else color,
                edgecolors=color if censored else "white",
                linewidths=1.2,
                zorder=3,
                label=row["family_label"] if row["family"] not in seen else None,
            )
            if row["family"] not in seen:
                seen.add(row["family"])
        if logx:
            ax.set_xscale("log")

    seen: set[str] = set()
    scatter(axes[0], "tau", True)
    axes[0].set_xlabel(r"Thời gian relaxation  $\tau = 1/(1-\lambda_2)$", fontsize=10)
    axes[0].set_ylabel("t₉₀ (bước)", fontsize=10)
    axes[0].set_title("(a) Theo thời gian trộn phổ", fontsize=11)
    _style_axes(axes[0])
    if seen:
        axes[0].legend(frameon=False, fontsize=8)

    seen.clear()
    scatter(axes[1], "path_length", False)
    if fit.get("slope") is not None:
        xs = np.linspace(0.0, max(r["path_length"] for r in rows) * 1.08 + 0.1, 60)
        axes[1].plot(
            xs, fit["intercept"] + fit["slope"] * xs, color="#3a3f47", linewidth=1.3
        )
        axes[1].annotate(
            "t₉₀ = %.2f + %.2f·L\nR² = %.3f" % (fit["intercept"], fit["slope"], fit["r2"]),
            xy=(0.04, 0.80),
            xycoords="axes fraction",
            fontsize=9,
        )
    axes[1].set_xlabel("Đường đi trung bình  L", fontsize=10)
    axes[1].set_ylabel("t₉₀ (bước)", fontsize=10)
    axes[1].set_title("(b) Theo khoảng cách thông tin", fontsize=11)
    _style_axes(axes[1])
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Job runner


@dataclass
class Job:
    id: str
    kind: str
    payload: dict[str, Any]
    status: str = "queued"
    progress: float = 0.0
    message: str = "Đang chờ..."
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str | None = None
    plots: dict[str, bytes] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        now = self.finished or time.time()
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "error": self.error,
            "elapsed": round(now - (self.started or self.created), 2),
            "result_ready": self.status == "done",
            "plots": sorted(self.plots) if self.status == "done" else [],
            "payload": self.payload,
        }


JOBS: "OrderedDict[str, Job]" = OrderedDict()
JOBS_LOCK = threading.Lock()
JOB_QUEUE: "queue.Queue[Job | None]" = queue.Queue()
WORKER_LOCK = threading.Lock()
WORKER: threading.Thread | None = None


def _ensure_worker() -> None:
    global WORKER
    with WORKER_LOCK:
        if WORKER is None or not WORKER.is_alive():
            WORKER = threading.Thread(target=_worker_loop, name="hive-job-worker", daemon=True)
            WORKER.start()


def _worker_loop() -> None:
    while True:
        job = JOB_QUEUE.get()
        if job is None:
            JOB_QUEUE.task_done()
            return
        try:
            _mutate(job, status="running", started=time.time(), message="Đang chạy...")
            result, plots = _dispatch(job)
            _mutate(
                job,
                status="done",
                progress=1.0,
                message="Hoàn tất",
                finished=time.time(),
                result=result,
                plots=plots,
            )
        except Exception as exc:  # surface the failure in the UI, keep server alive
            traceback.print_exc()
            _mutate(
                job,
                status="error",
                message="Lỗi",
                error=f"{type(exc).__name__}: {exc}",
                finished=time.time(),
            )
        finally:
            JOB_QUEUE.task_done()


def _mutate(job: Job, **changes: Any) -> None:
    with JOBS_LOCK:
        for key, value in changes.items():
            setattr(job, key, value)


def _progress(job: Job, fraction: float, message: str) -> None:
    _mutate(job, progress=max(0.0, min(1.0, fraction)), message=message)


def _dispatch(job: Job) -> tuple[dict[str, Any], dict[str, bytes]]:
    if job.kind == "reproduction":
        return _run_reproduction(job)
    if job.kind == "graph":
        return _run_graph(job)
    if job.kind == "compare":
        return _run_compare(job)
    if job.kind == "tests":
        return _run_tests(job)
    raise ValidationError(f"Loại job không hỗ trợ: {job.kind}")


def _run_reproduction(job: Job) -> tuple[dict[str, Any], dict[str, bytes]]:
    cfg = job.payload
    common = _common(cfg)
    curves = run_reproduction(
        n=common["n"],
        N=common["N"],
        steps=common["steps"],
        seeds=common["seeds"],
        scenario=common["scenario"],
        delta=common["delta"],
        progress=lambda done, total: _progress(
            job, 0.05 + 0.85 * done / total, f"Đang chạy seed {done}/{total}"
        ),
    )
    _progress(job, 0.93, "Đang tổng hợp kết quả...")

    means = {key: value.mean(axis=0) for key, value in curves.items()}
    stds = {key: value.std(axis=0) for key, value in curves.items()}
    finals = {key: float(value[:, -1].mean()) for key, value in curves.items()}
    hits = {key: t90(means[key]) for key in means}
    q = make_q(common["scenario"], common["n"])
    best = int(np.argmax(q))

    start = min(10, means["pmcl"].size - 1)
    max_dev = float(
        np.max(np.abs(means["pmcl"][start:] - means["mrd"][start:]))
    ) if means["pmcl"].size > start else 0.0

    result = {
        "experiment": "reproduction",
        "params": {**common, "best_arm": best, "q_best": float(q[best])},
        "means": means,
        "stds": stds,
        "finals": finals,
        "t90": {key: {"step": value[0], "hit": value[1]} for key, value in hits.items()},
        "max_deviation": max_dev,
    }
    _progress(job, 0.97, "Đang vẽ biểu đồ...")
    return result, {"curves": _plot_reproduction(curves)}


def _run_graph(job: Job) -> tuple[dict[str, Any], dict[str, bytes]]:
    cfg = job.payload
    common = _common(cfg)
    mode = "well_mixed" if cfg.get("mode") == "well_mixed" else "graph"
    family, param = _family_param(
        str(cfg.get("family", "ws")), cfg.get("param", 0.1), common["N"]
    )
    _progress(job, 0.04, "Đang dựng đồ thị...")
    adj, pos = build_graph(family, param, common["N"], seed=0)
    metrics = graph_metrics(adj)
    q = make_q(common["scenario"], common["n"])
    best = int(np.argmax(q))

    _progress(job, 0.08, "Đang mô phỏng...")
    res = simulate_on_graph(
        adj,
        q,
        steps=common["steps"],
        seeds=common["seeds"],
        delta=common["delta"],
        n_types=common["n"],
        mode=mode,
        return_state=True,
        progress=lambda done, total: _progress(
            job, 0.08 + 0.82 * done / total, f"Đang chạy seed {done}/{total}"
        ),
    )
    _progress(job, 0.94, "Đang tổng hợp kết quả...")

    mean_curve = res["curves"].mean(axis=0)
    std_curve = res["curves"].std(axis=0)
    finals = res["final"]
    summary = {
        "t90_median": float(np.median(res["t90"])),
        "t90_mean": float(np.mean(res["t90"])),
        "hit_rate": float(np.mean(res["hit"])),
        "final_median": float(np.median(finals)),
        "final95": float(np.mean(finals >= 0.95)),
        "consensus_median": float(np.median(res["consensus"])),
        "l2_median": float(np.median(res["l2_mrd"])),
    }
    final_types = np.asarray(res["final_types"], dtype=int)

    result = {
        "experiment": "graph",
        "family": family,
        "family_label": FAMILY_BY_ID[family]["label"],
        "param": param,
        "label": _config_label(family, param),
        "mode": mode,
        "params": {**common, "best_arm": best, "q_best": float(q[best])},
        "metrics": metrics,
        "summary": summary,
        "curves": {
            "mean": mean_curve,
            "std": std_curve,
            "mrd": res["mrd"],
        },
        "graph": {
            "layout": [list(pos[i]) for i in range(common["N"])],
            "types": final_types,
            "edges": _edges(adj)[:6000],
            "best": best,
        },
    }
    _progress(job, 0.97, "Đang vẽ biểu đồ...")
    plots = {
        "curves": _plot_graph_curves(mean_curve, std_curve, res["mrd"]),
        "graph": _plot_graph(adj, pos, final_types, best, common["n"]),
    }
    return result, plots


def _run_compare(job: Job) -> tuple[dict[str, Any], dict[str, bytes]]:
    cfg = job.payload
    common = _common(cfg)
    configs = cfg["configs"] if "configs" in cfg else DEFAULT_COMPARE
    if not isinstance(configs, list) or not configs:
        raise ValidationError("configs phải là danh sách không rỗng")
    if len(configs) > 14:
        raise ValidationError("tối đa 14 cấu hình mỗi lần so sánh")

    checked: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for item in configs:
        if not isinstance(item, dict):
            raise ValidationError("mỗi cấu hình phải là object {family, param}")
        family, param = _family_param(
            str(item.get("family", "")), item.get("param", 0), common["N"]
        )
        if (family, param) not in seen:
            seen.add((family, param))
            checked.append((family, param))

    q = make_q(common["scenario"], common["n"])
    best = int(np.argmax(q))
    rows: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}

    for index, (family, param) in enumerate(checked):
        adj, _ = build_graph(family, param, common["N"], seed=0)
        metrics = graph_metrics(adj)
        total_configs = len(checked)
        res = simulate_on_graph(
            adj,
            q,
            steps=common["steps"],
            seeds=common["seeds"],
            delta=common["delta"],
            n_types=common["n"],
            progress=lambda done, total, i=index: _progress(
                job,
                (i + done / total) / total_configs,
                f"Cấu hình {i + 1}/{total_configs}: seed {done}/{total}",
            ),
        )
        label = _config_label(family, param)
        rows.append(
            {
                "family": family,
                "family_label": FAMILY_BY_ID[family]["short"],
                "param": param,
                "label": label,
                "tau": metrics["tau"] if math.isfinite(metrics["tau"]) else None,
                "lambda2": metrics["lambda2"],
                "path_length": metrics["path_length"],
                "degree_mean": metrics["degree_mean"],
                "degree_cv": metrics["degree_cv"],
                "connected": metrics["connected"],
                "edges": metrics["edges"],
                "t90_median": float(np.median(res["t90"])),
                "t90_mean": float(np.mean(res["t90"])),
                "hit_rate": float(np.mean(res["hit"])),
                "final_median": float(np.median(res["final"])),
                "final95": float(np.mean(res["final"] >= 0.95)),
                "consensus_median": float(np.median(res["consensus"])),
                "l2_median": float(np.median(res["l2_mrd"])),
            }
        )
        curves[label] = {
            "mean": res["curves"].mean(axis=0),
            "std": res["curves"].std(axis=0),
        }

    fit_frame = pd.DataFrame(
        {
            "family": [row["family"] for row in rows],
            "param": [row["param"] for row in rows],
            "tau": [row["tau"] for row in rows],
            "path_length": [row["path_length"] for row in rows],
            "degree_mean": [row["degree_mean"] for row in rows],
            "degree_cv": [row["degree_cv"] for row in rows],
            "connected": [row["connected"] for row in rows],
            "hit_rate": [row["hit_rate"] for row in rows],
            "t90": [row["t90_median"] for row in rows],
        }
    )
    intercept, slope, r2, n_train = fit_law(fit_frame)
    diagnostics = law_diagnostics(fit_frame, n_boot=800)
    fit = {
        "intercept": None if not math.isfinite(intercept) else float(intercept),
        "slope": None if not math.isfinite(slope) else float(slope),
        "r2": None if not math.isfinite(r2) else float(r2),
        "n_train": int(n_train),
        "ci": diagnostics.get("ci", {}),
        "warnings": diagnostics.get("warnings", []),
        "influence": diagnostics.get("influence", [])[:3],
    }

    result = {
        "experiment": "compare",
        "params": {**common, "best_arm": best},
        "rows": rows,
        "fit": fit,
        "curves": curves,
    }
    _progress(job, 0.97, "Đang vẽ biểu đồ...")
    return result, {"compare": _plot_compare(rows, fit)}


def _run_tests(job: Job) -> tuple[dict[str, Any], dict[str, bytes]]:
    _progress(job, 0.1, "Đang chạy pytest...")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "experiment": "tests",
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output": output[-20000:],
    }, {}


def submit(payload: dict[str, Any]) -> Job:
    """Validate a request and queue it. Raises ``ValidationError`` on bad input."""
    if not isinstance(payload, dict):
        raise ValidationError("payload phải là object JSON")
    kind = str(payload.get("kind", ""))
    if kind not in {"reproduction", "graph", "compare", "tests"}:
        raise ValidationError("kind phải là reproduction, graph, compare hoặc tests")

    if kind in {"reproduction", "graph", "compare"}:
        common = _common(payload)
        if kind == "graph":
            _family_param(
                str(payload.get("family", "ws")),
                payload.get("param", 0.1),
                common["N"],
            )
        if kind == "compare":
            configs = payload["configs"] if "configs" in payload else DEFAULT_COMPARE
            if not isinstance(configs, list) or not configs:
                raise ValidationError("configs phải là danh sách không rỗng")
            if len(configs) > 14:
                raise ValidationError("tối đa 14 cấu hình mỗi lần so sánh")
            for item in configs:
                if not isinstance(item, dict):
                    raise ValidationError("mỗi cấu hình phải là object {family, param}")
                _family_param(
                    str(item.get("family", "")),
                    item.get("param", 0),
                    common["N"],
                )

    job = Job(id=uuid.uuid4().hex[:12], kind=kind, payload=payload)
    with JOBS_LOCK:
        JOBS[job.id] = job
        while len(JOBS) > JOB_HISTORY:
            oldest_id, oldest = next(iter(JOBS.items()))
            if oldest.status in {"queued", "running"}:
                break
            JOBS.pop(oldest_id)
    _ensure_worker()
    JOB_QUEUE.put(job)
    return job


# ---------------------------------------------------------------------------
# JSON helpers


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _stored_result(name: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{name}.csv"
    if name not in STORED_RESULTS or not path.exists():
        raise ValidationError(f"Không tìm thấy kết quả {name}.csv")
    frame = pd.read_csv(path)
    return {
        "name": name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "data": frame.head(500).to_dict(orient="records"),
        "truncated": len(frame) > 500,
        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
    }


# ---------------------------------------------------------------------------
# HTTP layer


class HiveHandler(BaseHTTPRequestHandler):
    server_version = "HiveMindRL/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if self.path.startswith("/api/"):
            super().log_message(format, *args)

    # -- helpers ------------------------------------------------------------

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(json_safe(payload), ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def error_json(self, status: int, message: str) -> None:
        self.send_json(status, {"error": message})

    def _path(self) -> str:
        return urlparse(self.path).path

    def _job(self, job_id: str) -> Job | None:
        with JOBS_LOCK:
            return JOBS.get(job_id)

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self._path()
        if path in {"/", "/index.html"}:
            return self._serve_file(UI_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/styles.css":
            return self._serve_file(UI_DIR / "styles.css", "text/css; charset=utf-8")
        if path == "/app.js":
            return self._serve_file(UI_DIR / "app.js", "text/javascript; charset=utf-8")
        if path == "/api/health":
            return self.send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "version": "0.1.0",
                    "root": str(ROOT),
                    "jobs": len(JOBS),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        if path == "/api/options":
            stored = {
                name: (RESULTS_DIR / f"{name}.csv").exists() for name in STORED_RESULTS
            }
            return self.send_json(
                HTTPStatus.OK,
                {
                    "families": FAMILIES,
                    "scenarios": sorted(SCENARIOS),
                    "presets": PRESETS,
                    "defaults": {
                        "N": 300,
                        "steps": 200,
                        "seeds": 20,
                        "n": 10,
                        "scenario": "middle",
                        "delta": 0.1,
                    },
                    "default_compare": DEFAULT_COMPARE,
                    "sweeps": {"ws_p": WS_P, "rgg_c": RGG_C, "bridge": BRIDGE_P},
                    "stored_results": stored,
                    "report_available": REPORT_PDF.exists(),
                    "heavy_tail_reference": {
                        "intercept": HEAVY_TAIL["intercept"],
                        "slope": HEAVY_TAIL["slope"],
                    },
                    "run_all": RUN_ALL,
                },
            )
        if path == "/api/law":
            diagnostic_path = RESULTS_DIR / "law_diagnostics.json"
            if diagnostic_path.exists():
                try:
                    payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                    return self.send_json(HTTPStatus.OK, payload)
                except (OSError, ValueError):
                    pass
            sweep_path = RESULTS_DIR / "graph_sweep.csv"
            if not sweep_path.exists():
                return self.error_json(
                    HTTPStatus.NOT_FOUND,
                    "Chưa có results/graph_sweep.csv — chạy scripts/run_all.py",
                )
            frame = config_summary(pd.read_csv(sweep_path))
            return self.send_json(
                HTTPStatus.OK, law_diagnostics(frame, n_boot=1000)
            )
        if path == "/report.pdf":
            if not REPORT_PDF.exists():
                return self.error_json(
                    HTTPStatus.NOT_FOUND, "Chưa có report/report.pdf — chạy scripts/build_report.py"
                )
            return self._serve_file(REPORT_PDF, "application/pdf")

        if path.startswith("/api/jobs/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) == 3:
                job = self._job(parts[2])
                if job is None:
                    return self.error_json(HTTPStatus.NOT_FOUND, "Job không tồn tại")
                return self.send_json(HTTPStatus.OK, job.public())
            if len(parts) == 4 and parts[3] == "result":
                job = self._job(parts[2])
                if job is None:
                    return self.error_json(HTTPStatus.NOT_FOUND, "Job không tồn tại")
                if job.status != "done":
                    return self.error_json(HTTPStatus.CONFLICT, "Job chưa hoàn tất")
                return self.send_json(HTTPStatus.OK, job.result)
            if len(parts) == 5 and parts[3] == "plots":
                job = self._job(parts[2])
                if job is None:
                    return self.error_json(HTTPStatus.NOT_FOUND, "Job không tồn tại")
                plot_name = parts[4].removesuffix(".png")
                plot = job.plots.get(plot_name)
                if plot is None:
                    return self.error_json(HTTPStatus.NOT_FOUND, "Không có biểu đồ này")
                return self._send(HTTPStatus.OK, plot, "image/png")

        if path == "/api/results":
            items = []
            for name in STORED_RESULTS:
                csv_path = RESULTS_DIR / f"{name}.csv"
                if csv_path.exists():
                    items.append(
                        {
                            "name": name,
                            "size": csv_path.stat().st_size,
                            "modified": time.strftime(
                                "%Y-%m-%d %H:%M",
                                time.localtime(csv_path.stat().st_mtime),
                            ),
                        }
                    )
            return self.send_json(
                HTTPStatus.OK,
                {"items": items, "report_available": REPORT_PDF.exists()},
            )
        if path.startswith("/api/results/") and path.endswith(".csv"):
            name = path.rsplit("/", 1)[-1].removesuffix(".csv")
            csv_path = RESULTS_DIR / f"{name}.csv"
            if name not in STORED_RESULTS or not csv_path.exists():
                return self.error_json(HTTPStatus.NOT_FOUND, f"Không có {name}.csv")
            return self._send(
                HTTPStatus.OK,
                csv_path.read_bytes(),
                "text/csv; charset=utf-8",
                {"Content-Disposition": f'attachment; filename="{name}.csv"'},
            )
        if path.startswith("/api/results/"):
            name = path.rsplit("/", 1)[-1].removesuffix(".csv")
            try:
                return self.send_json(HTTPStatus.OK, _stored_result(name))
            except ValidationError as exc:
                return self.error_json(HTTPStatus.NOT_FOUND, str(exc))
            except Exception as exc:  # malformed CSV should not kill the server
                return self.error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        if path.startswith("/api/"):
            return self.error_json(HTTPStatus.NOT_FOUND, "API không tồn tại")
        return self._send(
            HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8"
        )

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            return self.error_json(
                HTTPStatus.NOT_FOUND, f"Thiếu file giao diện: {path.name}"
            )
        return self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    # -- POST ---------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        path = self._path()
        if path != "/api/jobs":
            return self.error_json(HTTPStatus.NOT_FOUND, "API không tồn tại")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.error_json(HTTPStatus.BAD_REQUEST, "Content-Length không hợp lệ")
        if length <= 0 or length > MAX_BODY:
            return self.error_json(HTTPStatus.BAD_REQUEST, "Body rỗng hoặc quá lớn")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self.error_json(HTTPStatus.BAD_REQUEST, f"JSON không hợp lệ: {exc}")
        try:
            job = submit(payload)
        except ValidationError as exc:
            return self.error_json(HTTPStatus.BAD_REQUEST, str(exc))
        return self.send_json(HTTPStatus.ACCEPTED, job.public())


class HiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(host: str = "127.0.0.1", port: int = 8765) -> HiveHTTPServer:
    _ensure_worker()
    return HiveHTTPServer((host, port), HiveHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Giao diện mô phỏng HiveMindRL")
    parser.add_argument("--host", default="127.0.0.1", help="địa chỉ bind (mặc định 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="cổng (mặc định 8765)")
    parser.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    args = parser.parse_args(argv)

    # Windows consoles often default to cp1252, which cannot encode Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    server: HiveHTTPServer | None = None
    for offset in range(20):
        try:
            server = make_server(args.host, args.port + offset)
            break
        except OSError:
            continue
    if server is None:
        print(
            f"Không mở được cổng trong khoảng {args.port}–{args.port + 19}. "
            "Thử --port khác.",
            file=sys.stderr,
        )
        return 1

    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"HiveMindRL UI đang chạy tại {url}")
    print("Nhấn Ctrl+C để dừng.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
