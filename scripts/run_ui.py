"""Start the local dashboard for running and inspecting simulations.

Usage:
    python scripts/run_ui.py
    python scripts/run_ui.py --port 9000 --no-browser
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.webapp import main

if __name__ == "__main__":
    raise SystemExit(main())
