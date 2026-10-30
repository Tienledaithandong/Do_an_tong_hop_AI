import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_all_help_works_without_pythonpath():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--quick" in result.stdout
