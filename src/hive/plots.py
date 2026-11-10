"""Figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

COLORS = {
    "well_mixed": "#1f77b4",
    "pmcl": "#ff7f0e",
    "mrd": "#2ca02c",
    "two_room": "#2ca02c",
    "ws": "#1f77b4",
    "rgg": "#d62728",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_reproduction(curves: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    steps = curves["well_mixed"].shape[1]
    t = np.arange(steps)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    styles = {
        "well_mixed": ("Bầy đàn (weighted voter)", "-", COLORS["well_mixed"]),
        "pmcl": ("P-MCL (tác tử đơn)", "-", COLORS["pmcl"]),
        "mrd": ("MRD (rời rạc δ = 1)", "--", COLORS["mrd"]),
    }
    for key, (label, line_style, color) in styles.items():
        mean = curves[key].mean(axis=0)
        std = curves[key].std(axis=0)
        ax.plot(t, mean, line_style, color=color, label=label, linewidth=1.8)
        if key != "mrd":
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_xlabel("Bước quyết định")
    ax.set_ylabel("Tỉ lệ chọn phương án tốt nhất")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, steps - 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "fig1_reproduction")


def config_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, param), group in df.groupby(["family", "param"]):
        rows.append(
            {
                "family": family,
                "param": param,
                "tau": group["tau"].iloc[0],
                "path_length": group["path_length"].iloc[0],
                "t90": group["t90"].median(),
                "hit_rate": group["t90_hit"].mean(),
                "final": group["final_best_mass"].median(),
                "final95": (group["final_best_mass"] >= 0.95).mean(),
                "consensus": group["consensus"].median(),
                "degree_mean": group["degree_mean"].iloc[0],
                "degree_cv": group["degree_cv"].iloc[0],
                "connected": bool(np.isfinite(group["tau"].iloc[0])),
            }
        )
    summary = pd.DataFrame(rows)
    # A rewiring probability too small to touch the lattice reproduces the
    # same graph (identical tau and L). Keep one row per distinct topology
    # so the fit and the figure do not count one configuration twice.
    key = summary[["tau", "path_length"]].round(6).astype(str).agg("|".join, axis=1)
    return summary.loc[~key.duplicated()].reset_index(drop=True)


def fit_law(summary: pd.DataFrame) -> tuple[float, float, float, int]:
    """Fit ``t90 = intercept + slope * L`` on non-intervention families.

    Two-room graphs are the held-out intervention. Disconnected graphs and
    configurations whose ``t90`` is censored in more than half of the seeds
    are excluded when the corresponding columns are present; otherwise the
    fit would mix a mixing-time proxy with a travel-time proxy.
    Returns ``(intercept, slope, r2, n_train)``.
    """
    train = summary[summary["family"] != "two_room"]
    if "connected" in train:
        train = train[train["connected"].astype(bool)]
    if "tau" in train:
        train = train[np.isfinite(pd.to_numeric(train["tau"], errors="coerce"))]
    if "hit_rate" in train:
        train = train[train["hit_rate"] >= 0.5]
    train = train[
        np.isfinite(pd.to_numeric(train["path_length"], errors="coerce"))
        & np.isfinite(pd.to_numeric(train["t90"], errors="coerce"))
    ]
    x = train["path_length"].to_numpy(float)
    y = train["t90"].to_numpy(float)
    if x.size < 3:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    r2 = 1.0 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(intercept), float(slope), float(r2), int(x.size)


def plot_law(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    summary = config_summary(df)
    summary = summary[np.isfinite(summary["tau"])]
    measured = summary[summary["hit_rate"] >= 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    labels = {"ws": "WS", "rgg": "RGG", "two_room": "Hai buồng (kiểm định)"}
    ax = axes[0]
    for family in ("ws", "rgg", "two_room"):
        sub = measured[measured["family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["tau"], sub["t90"], s=40, color=COLORS[family], label=labels[family]
        )
        if len(sub) >= 3:
            slope, intercept = np.polyfit(
                np.log10(sub["tau"]), sub["t90"], 1
            )
            xs = np.logspace(
                np.log10(sub["tau"].min()), np.log10(sub["tau"].max()), 40
            )
            ax.plot(xs, intercept + slope * np.log10(xs), color=COLORS[family],
                    linewidth=1.0, alpha=0.7)
    censored = summary[summary["hit_rate"] < 0.5]
    if not censored.empty:
        ax.scatter(
            censored["tau"],
            censored["t90"],
            s=40,
            facecolors="none",
            edgecolors="0.4",
            label="bị censor (<50% seed)",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Thời gian relaxation  $\tau = 1/(1-\lambda_2)$")
    ax.set_ylabel("Số bước đạt 90% phương án tốt nhất")
    ax.set_title("(a) Thời gian relaxation: hai xu hướng tách nhau")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for family in ("ws", "rgg", "two_room"):
        sub = measured[measured["family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["path_length"],
            sub["t90"],
            s=40,
            color=COLORS[family],
            label=labels[family],
        )
    # Calibrate on the non-intervention configurations and keep the two-room
    # graphs as a held-out check, so the line is not fitted to its test points.
    intercept, slope, r2, n_train = fit_law(measured)
    if n_train >= 3:
        xs = np.linspace(0.0, measured["path_length"].max() * 1.05, 50)
        ax.plot(xs, intercept + slope * xs, color="0.2", linewidth=1.2)
        ax.annotate(
            " $t_{90}$ = %.2f + %.2f $\\cdot$ L\n $R^2$ = %.2f  (fit %d cấu hình)"
            % (intercept, slope, r2, n_train),
            xy=(0.04, 0.80),
            xycoords="axes fraction",
            fontsize=10,
        )
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Đường đi trung bình  $L$")
    ax.set_ylabel("Số bước đạt 90% phương án tốt nhất")
    ax.set_title("(b) Đường đi trung bình dự báo cả hai họ")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir, "fig2_law")


def plot_graph_seeds(
    graphs: pd.DataFrame, law: dict, out_dir: Path, suffix: str = ""
) -> list[Path]:
    """Config-level law with graph-to-graph spread across redrawn graphs.

    Each point averages the redrawn graphs of one configuration; the error
    bars are the standard deviation across graphs, so the figure shows how
    much of the scatter is topology sampling rather than swarm-seed noise.
    """
    usable = graphs[graphs["hit_rate"] >= 0.5]
    agg = (
        usable.groupby(["config", "family"])
        .agg(
            L=("path_length", "mean"),
            L_sd=("path_length", "std"),
            t=("t90_median", "mean"),
            t_sd=("t90_median", "std"),
        )
        .reset_index()
    )
    agg[["L_sd", "t_sd"]] = agg[["L_sd", "t_sd"]].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    labels = {"complete": "Đầy đủ", "ws": "WS", "rgg": "RGG"}
    colors = {"complete": COLORS["well_mixed"], "ws": COLORS["ws"], "rgg": COLORS["rgg"]}
    for family in ("complete", "ws", "rgg"):
        sub = agg[agg["family"] == family]
        if sub.empty:
            continue
        ax.errorbar(
            sub["L"], sub["t"], xerr=sub["L_sd"], yerr=sub["t_sd"],
            fmt="o", ms=5, color=colors[family], label=labels[family],
            elinewidth=1.0, capsize=2, alpha=0.85,
        )
    if "slope" in law:
        xs = np.linspace(0.0, agg["L"].max() * 1.05, 50)
        ax.plot(xs, law["intercept"] + law["slope"] * xs, color="0.2", linewidth=1.2)
        ci = law["ci"]["slope"]
        ax.annotate(
            " $t_{90}$ = %.2f + %.2f $\\cdot$ L\n $R^2$ = %.2f\n slope 95%% CI [%.2f, %.2f]"
            % (law["intercept"], law["slope"], law["r2"], ci[0], ci[1]),
            xy=(0.04, 0.72), xycoords="axes fraction", fontsize=9,
        )
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Đường đi trung bình  $L$  (trung bình $\pm$ SD qua các graph)")
    ax.set_ylabel("Số bước đạt 90% (trung bình median mỗi graph)")
    ax.set_title("Quy luật L hiệu chuẩn trên nhiều graph seed")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig8_graph_seeds{suffix}")


def plot_trajectories(
    mean_curves: dict[str, np.ndarray], mrd: np.ndarray, out_dir: Path
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    keys = [
        ("complete:1", "Đồ thị đầy đủ", "0.15", "-"),
        ("ws:1", r"WS $p=1$", "#1f77b4", "-"),
        ("ws:0.1", r"WS $p=0{,}1$", "#ff7f0e", "-"),
        ("ws:0.05", r"WS $p=0{,}05$", "#9467bd", "-"),
        ("ws:0.01", r"WS $p=0{,}01$", "#d62728", "-"),
        ("ws:0", r"WS $p=0$ (vòng lattice)", "#8c564b", "-"),
    ]
    for key, label, color, style in keys:
        lookup = key if key in mean_curves else key.replace(":", "_")
        if lookup not in mean_curves:
            continue
        curve = mean_curves[lookup]
        ax.plot(
            np.arange(curve.size), curve, style, color=color, label=label, linewidth=1.6
        )
    ax.plot(
        np.arange(mrd.size),
        mrd,
        "--",
        color=COLORS["mrd"],
        linewidth=1.4,
        label="MRD (lý thuyết trộn đều)",
    )
    ax.set_xlabel("Bước quyết định")
    ax.set_ylabel("Tỉ lệ chọn phương án tốt nhất")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "fig3_trajectories")


def _median_ci(values: np.ndarray, n_boot: int = 800, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    median = float(np.median(values))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    boots = np.median(draws, axis=1)
    return median, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def plot_two_room_source(
    per_seed: pd.DataFrame,
    curves: dict[int, np.ndarray],
    out_dir: Path,
) -> list[Path]:
    """Target-room arrival times when only the source room is seeded."""
    bridges = sorted(curves)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bridges)))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    for color, bridge in zip(colors, bridges):
        curve = np.asarray(curves[bridge], dtype=float)
        ax.plot(
            np.arange(curve.size),
            curve,
            color=color,
            linewidth=1.8,
            label=f"cầu = {bridge}",
        )
    ax.set_xlabel("Bước quyết định")
    ax.set_ylabel("Tỉ lệ phương án tốt trong buồng đích")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a) Lan truyền sang buồng đích")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for offset, metric, marker, color in (
        (-0.12, "t50_target", "o", "#0e7c86"),
        (0.12, "t90_target", "s", "#d97706"),
    ):
        hit_column = "hit50_target" if metric == "t50_target" else "hit90_target"
        xs: list[float] = []
        medians: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for bridge in bridges:
            mask = (per_seed["bridge"] == bridge) & per_seed[hit_column]
            values = per_seed.loc[mask, metric].to_numpy(float)
            median, lo, hi = _median_ci(values)
            xs.append(bridge + offset)
            medians.append(median)
            lower.append(median - lo)
            upper.append(hi - median)
        ax.errorbar(
            xs,
            medians,
            yerr=np.vstack([lower, upper]),
            fmt=marker,
            color=color,
            capsize=3,
            linewidth=1.2,
            label="50% buồng đích" if metric == "t50_target" else "90% buồng đích",
        )
    ax.set_xlabel("Số cạnh cầu nối hai buồng")
    ax.set_ylabel("Số bước (trung vị, CI 95%)")
    ax.set_title("(b) Thời gian qua cầu")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir, "fig4_two_room_source")


def plot_two_room_replicates(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Wait at the bridge vs conductance, and the split of the delay."""
    sizes = sorted(summary["N"].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    for color, N in zip(colors, sizes):
        sub = summary[summary["N"] == N]
        for key, style in (("t_first", "o-"), ("t50", "s--")):
            ok = sub[np.isfinite(sub[f"{key}_median"])]
            ax.plot(
                ok["phi"],
                ok[f"{key}_median"],
                style,
                color=color,
                label=f"N={N}, {'lần đầu tới' if key == 't_first' else 't50'}",
                markersize=4,
                alpha=0.9 if key == "t_first" else 0.6,
            )
            # Bands are dropped where the upper bound is censored (inf).
            lo = ok[f"{key}_lo"].to_numpy(dtype=float)
            hi = ok[f"{key}_hi"].to_numpy(dtype=float)
            hi = np.where(np.isfinite(hi), hi, np.nan)
            ax.fill_between(
                ok["phi"].to_numpy(),
                np.minimum(lo, ok[f"{key}_median"].to_numpy()),
                hi,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"độ dẫn của cầu $\phi$ = cầu / (N/2)")
    ax.set_ylabel("trung vị số bước")
    ax.set_title("(a) Thời gian chờ ở cầu theo độ dẫn")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    width = 0.8 / len(sizes)
    bridges = sorted(summary["bridge"].unique())
    x = np.arange(len(bridges))
    for k, (color, N) in enumerate(zip(colors, sizes)):
        sub = summary[summary["N"] == N].set_index("bridge").reindex(bridges)
        first = sub["t_first_median"].to_numpy(dtype=float)
        total = sub["t50_median"].to_numpy(dtype=float)
        pos = x - 0.4 + width * (k + 0.5)
        ax.bar(pos, first, width * 0.95, color=color, label=f"N={N}: chờ ở cầu")
        ax.bar(
            pos,
            np.where(np.isfinite(total), total - first, 0.0),
            width * 0.95,
            bottom=first,
            color=color,
            alpha=0.4,
            label=f"N={N}: lan trong buồng",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bridges])
    ax.set_xlabel("độ rộng cầu (số cạnh)")
    ax.set_ylabel("trung vị số bước tới t50")
    ax.set_title("(b) Tách độ trễ: chờ ở cầu và lan trong buồng")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return _save(fig, out_dir, "fig5_two_room_replicates")


def plot_two_room_sparse(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Wait at the bridge against each candidate predictor, clique and sparse rooms."""
    sizes = sorted(summary["N"].unique())
    degrees = sorted(summary["room_degree"].unique())
    colors = dict(zip(sizes, plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))))
    markers = dict(zip(degrees, ["o", "s", "^", "D"]))
    panels = [
        ("phi", r"bridge / (N/2)", "(a) theo số cạnh cầu trên mỗi nút"),
        ("tau", r"$\tau$", r"(b) theo thời gian relaxation $\tau$"),
        ("phi_deg", "bridge / bậc phòng", "(c) theo số cạnh cầu trên mỗi hàng xóm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (key, label, title) in zip(axes, panels):
        for degree in degrees:
            for N in sizes:
                sub = summary[(summary["room_degree"] == degree) & (summary["N"] == N)]
                sub = sub[np.isfinite(sub["t_first_median"])]
                if sub.empty:
                    continue
                ax.scatter(
                    sub[key],
                    sub["t_first_median"],
                    marker=markers[degree],
                    color=colors[N],
                    s=34,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.4,
                    label=f"N={N}, {'clique' if degree == 0 else f'd={degree}'}",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("trung vị `t_first` (bước)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, fontsize=6.3, ncol=2, loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_two_room_sparse")


def plot_two_room_hetero(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Wait at the bridge by endpoint placement, and against two predictors."""
    ba = summary[summary["kind"] == "ba"]
    target_colors = {"leaf": "#2a9d8f", "random": "#6c757d", "hub": "#d1495b"}
    source_styles = {"leaf": ":", "random": "-", "hub": "--"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    for (source, target), g in ba.groupby(["source", "target"]):
        g = g[np.isfinite(g["t_first_median"])].sort_values("bridge")
        ax.plot(
            g["bridge"],
            g["t_first_median"],
            linestyle=source_styles[source],
            marker="o",
            markersize=3.5,
            color=target_colors[target],
            alpha=0.9,
            label=f"nguồn {source} → đích {target}",
        )
    reg = summary[summary["kind"] == "regular"].sort_values("bridge")
    if not reg.empty:
        ax.plot(reg["bridge"], reg["t_first_median"], "k-", marker="s", markersize=3.5,
                linewidth=2, label="đối chứng: đều bậc 6")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("độ rộng cầu (số cạnh)")
    ax.set_ylabel("trung vị `t_first` (bước)")
    ax.set_title("(a) Vị trí đầu cầu: màu = phía đích, nét = phía nguồn", fontsize=9)
    ax.legend(fontsize=5.8, ncol=2)
    ax.grid(alpha=0.3, which="both")

    for ax, (key, label, title) in zip(
        axes[1:],
        [
            ("inv_end_target", r"$\sum 1/\deg$(đầu cầu phía đích)", "(b) theo bậc của đầu cầu đích"),
            ("phi_deg_room", "cầu / bậc trung bình của buồng đích", "(c) theo bậc trung bình buồng"),
        ],
    ):
        for kind, marker in (("ba", "o"), ("regular", "s")):
            for target, color in target_colors.items():
                sub = summary[(summary["kind"] == kind) & (summary["target"] == target)]
                sub = sub[np.isfinite(sub["t_first_median"])]
                if sub.empty:
                    continue
                ax.scatter(sub[key], sub["t_first_median"], marker=marker, color=color,
                           s=26, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_title(title, fontsize=9)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_two_room_hetero")
