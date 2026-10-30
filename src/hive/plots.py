"""Figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
                "degree_cv": group["degree_cv"].iloc[0],
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

    The two-room graphs are the held-out intervention, so they are excluded
    from the calibration. Returns ``(intercept, slope, r2, n_train)``.
    """
    train = summary[summary["family"] != "two_room"]
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
