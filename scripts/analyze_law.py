"""Recompute the empirical law with uncertainty and influence diagnostics.

Usage:
    python scripts/analyze_law.py
    python scripts/analyze_law.py --bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.analysis import law_diagnostics
from hive.plots import config_summary
from hive.provenance import write_metadata

RESULTS = ROOT / "results"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--input", default=str(RESULTS / "graph_sweep.csv"))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Không thấy {path}; chạy scripts/run_all.py trước.")
    df = pd.read_csv(path)
    summary = config_summary(df)
    diag = law_diagnostics(summary, n_boot=args.bootstrap)
    diag["source"] = path.name
    out = path.parent / "law_diagnostics.json"
    out.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metadata(
        out.parent, "analysis", {"n_boot": args.bootstrap, "bootstrap_seed": 0},
        name="law_diagnostics_metadata", inputs=[path], artifacts=[out],
    )

    print(f"n = {diag['n']} cấu hình")
    print(f"t90 = {diag['intercept']:.2f} + {diag['slope']:.2f} * L, R² = {diag['r2']:.3f}")
    ci = diag.get("ci", {})
    if ci:
        print(
            "CI 95%: intercept "
            f"[{ci['intercept'][0]:.2f}, {ci['intercept'][1]:.2f}], "
            f"slope [{ci['slope'][0]:.2f}, {ci['slope'][1]:.2f}], "
            f"dự đoán tại L=2 [{ci['prediction_at_L2'][0]:.2f}, "
            f"{ci['prediction_at_L2'][1]:.2f}]"
        )
    loo = diag.get("leave_one_out", {})
    if loo:
        print(
            "Leave-one-out: slope "
            f"[{loo['slope_min']:.2f}, {loo['slope_max']:.2f}], "
            f"R² [{loo['r2_min']:.3f}, {loo['r2_max']:.3f}]"
        )
    print("Ảnh hưởng lớn nhất:")
    for row in diag.get("influence", [])[:3]:
        print(
            f"  {row['label']:>12}  Cook's D = {row['cook']:.2f}  "
            f"leverage = {row['leverage']:.2f}"
        )
    for warning in diag.get("warnings", []):
        print(f"  ! {warning}")
    print("  wrote", out)


if __name__ == "__main__":
    main()
