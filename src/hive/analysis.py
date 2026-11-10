"""Diagnostics for the empirical ``t90 ~ L`` law.

The report quotes a single least-squares line. A single line is not enough:
with 16 configurations, two long-lattice points carry most of the slope, and
WS/RGG differ by family. This module reports the uncertainty and influence
that the headline number hides.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

L_REF = 2.0


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def law_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows used to calibrate the law: connected, non-intervention, uncensored."""
    train = summary[summary["family"] != "two_room"].copy()
    if "connected" in train:
        train = train[train["connected"].astype(bool)]
    if "tau" in train:
        train = train[np.isfinite(_as_float(train["tau"]))]
    if "hit_rate" in train:
        train = train[train["hit_rate"] >= 0.5]
    train = train[
        np.isfinite(_as_float(train["path_length"]))
        & np.isfinite(_as_float(train["t90"]))
    ]
    return train.reset_index(drop=True)


def _ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    ss_res = float(resid @ resid)
    centered = y - y.mean()
    ss_tot = float(centered @ centered)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = n - p
    if dof > 0:
        mse = ss_res / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        cov = mse * xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            tvalues = beta / se
        pvalues = 2.0 * stats.t.sf(np.abs(tvalues), dof)
    else:
        mse = float("nan")
        xtx_inv = np.full((p, p), np.nan)
        se = np.full(p, np.nan)
        pvalues = np.full(p, np.nan)
    return {
        "beta": beta,
        "se": se,
        "pvalues": pvalues,
        "resid": resid,
        "r2": r2,
        "mse": mse,
        "xtx_inv": xtx_inv,
        "n": n,
        "p": p,
    }


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    model = _ols(y, X)
    return float(model["beta"][0]), float(model["beta"][1]), float(model["r2"])


def bootstrap_law(
    train: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    x = _as_float(train["path_length"])
    y = _as_float(train["t90"])
    if x.size < 3:
        return {}
    rng = np.random.default_rng(seed)
    intercepts = np.empty(n_boot)
    slopes = np.empty(n_boot)
    predictions = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        intercept, slope, _ = _fit_line(x[idx], y[idx])
        intercepts[i] = intercept
        slopes[i] = slope
        predictions[i] = intercept + slope * L_REF
    q = lambda draws: [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return {
        "intercept": q(intercepts),
        "slope": q(slopes),
        "prediction_at_L2": q(predictions),
    }


def _collapse_graphs(
    graphs: list[tuple[float, np.ndarray, np.ndarray]],
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """One (L, t90) point from a config's graphs: mean L, mean of per-graph medians.

    Each graph is ``(L, t90, hit)``. The per-graph median is taken over the
    *hit* seeds only, so a censoring sentinel never enters as an observed time.
    With ``rng`` the (t90, hit) pairs inside each graph are resampled together
    first, so censoring is handled inside every bootstrap replicate; a resample
    with no hit seeds contributes no median and is left out of that replicate's
    config mean (NaN if the whole config has none).
    """
    ls: list[float] = []
    ts: list[float] = []
    for length, t90s, hits in graphs:
        if rng is not None:
            idx = rng.integers(0, t90s.size, t90s.size)
            t90s, hits = t90s[idx], hits[idx]
        reached = t90s[hits]
        ls.append(length)
        ts.append(float(np.median(reached)) if reached.size else float("nan"))
    finite = [t for t in ts if np.isfinite(t)]
    return float(np.mean(ls)), float(np.mean(finite)) if finite else float("nan")


def hierarchical_bootstrap_law(
    seeds: pd.DataFrame, n_boot: int = 500, seed: int = 0
) -> dict[str, Any]:
    """Cluster bootstrap for the ``t90 ~ L`` law over graphs then seeds.

    ``seeds`` has one row per (config, graph, swarm seed) with columns
    ``config``, ``graph_seed``, ``path_length``, ``t90`` and ``t90_hit``. The
    point estimate collapses each config to one (mean L, mean per-graph
    hit-only median t90) point and fits OLS. Each bootstrap replicate resamples
    graphs with replacement within every config and (t90, hit) pairs within each
    drawn graph, then refits, so the CI reflects both levels of variation and
    censoring is handled inside each resample. Replicates whose resamples leave
    a config with no hit seeds are skipped rather than averaged with sentinels.

    Note this CI conditions on the *fixed* set of configurations: it does not
    reflect sensitivity to adding/removing configs or graph families (see
    ``leave_one_family_out`` and the config-level ``bootstrap_law`` for that).
    """
    groups: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}
    for config, cdf in seeds.groupby("config", sort=True):
        graphs: list[tuple[float, np.ndarray, np.ndarray]] = []
        for _, gdf in cdf.groupby("graph_seed", sort=True):
            graphs.append(
                (
                    float(gdf["path_length"].iloc[0]),
                    _as_float(gdf["t90"]),
                    gdf["t90_hit"].to_numpy(dtype=bool),
                )
            )
        groups[str(config)] = graphs
    configs = list(groups)
    if len(configs) < 3:
        return {"n_configs": len(configs), "error": "cần ít nhất 3 cấu hình"}

    x = np.array([_collapse_graphs(groups[c])[0] for c in configs])
    y = np.array([_collapse_graphs(groups[c])[1] for c in configs])
    intercept, slope, r2 = _fit_line(x, y)

    rng = np.random.default_rng(seed)
    intercepts: list[float] = []
    slopes: list[float] = []
    r2s: list[float] = []
    predictions: list[float] = []
    for _ in range(n_boot):
        xb = np.empty(len(configs))
        yb = np.empty(len(configs))
        for j, c in enumerate(configs):
            graphs = groups[c]
            g = len(graphs)
            drawn = [graphs[i] for i in rng.integers(0, g, g)]
            xb[j], yb[j] = _collapse_graphs(drawn, rng)
        if not (np.all(np.isfinite(xb)) and np.all(np.isfinite(yb))):
            continue
        i_b, s_b, r_b = _fit_line(xb, yb)
        intercepts.append(i_b)
        slopes.append(s_b)
        r2s.append(r_b)
        predictions.append(i_b + s_b * L_REF)
    if not slopes:
        return {"n_configs": len(configs), "error": "không mẫu bootstrap nào đủ seed đạt"}

    q = lambda draws: [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return {
        "n_configs": len(configs),
        "n_graphs": int(sum(len(g) for g in groups.values())),
        "n_seeds": int(len(seeds)),
        "n_boot_used": len(slopes),
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "prediction_at_L2": float(intercept + slope * L_REF),
        "ci": {
            "intercept": q(intercepts),
            "slope": q(slopes),
            "r2": q(r2s),
            "prediction_at_L2": q(predictions),
        },
    }


def _label(row: pd.Series) -> str:
    family = str(row.get("family", "?"))
    param = row.get("param", float("nan"))
    try:
        return f"{family}:{float(param):g}"
    except (TypeError, ValueError):
        return family


def influence(train: pd.DataFrame) -> list[dict[str, Any]]:
    x = _as_float(train["path_length"])
    y = _as_float(train["t90"])
    X = np.column_stack([np.ones_like(x), x])
    model = _ols(y, X)
    h = np.einsum("ij,jk,ik->i", X, model["xtx_inv"], X)
    p = X.shape[1]
    mse = model["mse"]
    if np.isfinite(mse) and mse > 0:
        cook = (model["resid"] ** 2) / (p * mse) * h / np.clip(1.0 - h, 1e-12, None) ** 2
    else:
        cook = np.full_like(x, np.nan)
    order = np.argsort(np.nan_to_num(cook, nan=-1.0))[::-1]
    rows = []
    for i in order[:5]:
        rows.append(
            {
                "label": _label(train.iloc[i]),
                "cook": float(cook[i]) if np.isfinite(cook[i]) else None,
                "leverage": float(h[i]),
                "residual": float(model["resid"][i]),
                "L": float(x[i]),
                "t90": float(y[i]),
            }
        )
    return rows


def leave_one_out(train: pd.DataFrame) -> dict[str, Any]:
    x = _as_float(train["path_length"])
    y = _as_float(train["t90"])
    if x.size < 4:
        return {}
    slopes: list[float] = []
    r2s: list[float] = []
    predictions: list[float] = []
    for i in range(x.size):
        mask = np.arange(x.size) != i
        intercept, slope, r2 = _fit_line(x[mask], y[mask])
        slopes.append(slope)
        r2s.append(r2)
        predictions.append(intercept + slope * L_REF)
    return {
        "slope_min": float(np.nanmin(slopes)),
        "slope_max": float(np.nanmax(slopes)),
        "r2_min": float(np.nanmin(r2s)),
        "r2_max": float(np.nanmax(r2s)),
        "prediction_at_L2_min": float(np.nanmin(predictions)),
        "prediction_at_L2_max": float(np.nanmax(predictions)),
    }


def leave_one_family_out(train: pd.DataFrame) -> list[dict[str, Any]]:
    """Refit the law after dropping each graph family in turn.

    The hierarchical bootstrap conditions on the fixed config set, so it cannot
    show how much the slope depends on *which* families are present. This
    sensitivity fills that gap: a large swing when one family is removed means
    the headline slope is partly a between-family contrast, not a within-family
    law.
    """
    out: list[dict[str, Any]] = []
    for family in sorted(train["family"].unique()):
        sub = train[train["family"] != family]
        x = _as_float(sub["path_length"])
        y = _as_float(sub["t90"])
        intercept, slope, r2 = _fit_line(x, y)
        out.append(
            {
                "dropped": str(family),
                "n": int(x.size),
                "intercept": float(intercept),
                "slope": float(slope),
                "r2": float(r2),
            }
        )
    return out


def family_effect(train: pd.DataFrame) -> dict[str, Any]:
    """Additive and interaction models with RGG/complete as the baseline."""
    x = _as_float(train["path_length"])
    y = _as_float(train["t90"])
    is_ws = (train["family"] == "ws").to_numpy(dtype=float)
    is_complete = (train["family"] == "complete").to_numpy(dtype=float)
    if x.size < 6:
        return {}
    X_add = np.column_stack([np.ones_like(x), x, is_ws, is_complete])
    add = _ols(y, X_add)
    X_int = np.column_stack([np.ones_like(x), x, is_ws, is_complete, is_ws * x])
    inter = _ols(y, X_int)
    return {
        "ws_offset": float(add["beta"][2]),
        "ws_offset_se": float(add["se"][2]),
        "ws_offset_p": float(add["pvalues"][2]),
        "complete_offset": float(add["beta"][3]),
        "additive_r2": float(add["r2"]),
        "interaction_slope": float(inter["beta"][4]),
        "interaction_p": float(inter["pvalues"][4]),
        "interaction_r2": float(inter["r2"]),
    }


def _model_row(name: str, y: np.ndarray, columns: list[np.ndarray]) -> dict[str, Any]:
    X = np.column_stack([np.ones_like(y), *columns])
    model = _ols(y, X)
    n, p = X.shape
    adjusted = 1.0 - (1.0 - model["r2"]) * (n - 1) / (n - p) if n > p else float("nan")
    return {"name": name, "r2": float(model["r2"]), "adjusted_r2": float(adjusted)}


def degree_models(train: pd.DataFrame) -> list[dict[str, Any]]:
    if "degree_mean" not in train or "degree_cv" not in train:
        return []
    x = _as_float(train["path_length"])
    y = _as_float(train["t90"])
    degree = _as_float(train["degree_mean"])
    cv = _as_float(train["degree_cv"])
    if not (np.all(np.isfinite(degree)) and np.all(degree > 0)):
        return []
    rows = [_model_row("L", y, [x])]
    rows.append(_model_row("L + log(degree)", y, [x, np.log(degree)]))
    rows.append(_model_row("L + degree CV", y, [x, cv]))
    rows.append(_model_row("L + log(degree) + CV", y, [x, np.log(degree), cv]))
    return rows


def correlations(train: pd.DataFrame) -> dict[str, Any]:
    y = _as_float(train["t90"])
    out: dict[str, Any] = {}
    for column in ("tau", "path_length", "degree_mean", "degree_cv"):
        if column not in train:
            continue
        x = _as_float(train[column])
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        pearson = stats.pearsonr(x[mask], y[mask])
        spearman = stats.spearmanr(x[mask], y[mask])
        out[column] = {
            "pearson": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "n": int(mask.sum()),
        }
    return out


def law_diagnostics(
    summary: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    train = law_frame(summary)
    x = _as_float(train["path_length"])
    y = _as_float(train["t90"])
    if x.size < 3:
        return {"n": int(x.size), "error": "cần ít nhất 3 cấu hình để hiệu chuẩn"}
    intercept, slope, r2 = _fit_line(x, y)
    boot = bootstrap_law(train, n_boot=n_boot, seed=seed)
    infl = influence(train)
    loo = leave_one_out(train)
    lof = leave_one_family_out(train)
    family = family_effect(train)
    degree = degree_models(train)
    corr = correlations(train)

    warnings: list[str] = []
    max_cook = max((row["cook"] or 0.0) for row in infl) if infl else 0.0
    if max_cook > 1.0:
        warnings.append(
            f"Một cấu hình có Cook's D = {max_cook:.1f} (> 1), chi phối hệ số hồi quy."
        )
    if loo and np.isfinite(loo["slope_min"]) and slope:
        ratio = (loo["slope_max"] - loo["slope_min"]) / abs(slope)
        if ratio > 0.5:
            warnings.append(
                "Bỏ một cấu hình làm hệ số góc đổi "
                f"{100 * ratio:.0f}% — quy luật chưa ổn định."
            )
    fam_slopes = [r["slope"] for r in lof if np.isfinite(r["slope"])]
    if fam_slopes and slope:
        swing = (max(fam_slopes) - min(fam_slopes)) / abs(slope)
        if swing > 0.5:
            warnings.append(
                "Bỏ một họ đồ thị làm hệ số góc đổi "
                f"{100 * swing:.0f}% — độ dốc phụ thuộc tập họ đang có."
            )
    if family and np.isfinite(family["ws_offset_p"]) and family["ws_offset_p"] < 0.05:
        warnings.append(
            "Có hiệu ứng họ đồ thị: WS lệch "
            f"{family['ws_offset']:+.1f} bước so với baseline (p = {family['ws_offset_p']:.3f})."
        )
    if degree:
        gain = max(row["r2"] for row in degree) - degree[0]["r2"]
        if gain > 0.02:
            warnings.append(
                "Bậc trung bình / degree CV cũng giải thích thêm "
                f"{100 * gain:.1f}% phương sai; L chưa được tách khỏi bậc."
            )
    if boot:
        width = boot["slope"][1] - boot["slope"][0]
        if width > 2.0:
            warnings.append(
                f"CI 95% của hệ số góc rộng ({boot['slope'][0]:.2f}…{boot['slope'][1]:.2f}); "
                "không nên đọc là hằng số."
            )

    return {
        "n": int(x.size),
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "prediction_at_L2": float(intercept + slope * L_REF),
        "ci": boot,
        "leave_one_out": loo,
        "leave_one_family": lof,
        "influence": infl,
        "family_effect": family,
        "degree_models": degree,
        "correlations": corr,
        "warnings": warnings,
    }
