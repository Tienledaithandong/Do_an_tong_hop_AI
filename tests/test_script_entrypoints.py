import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script,flag",
    [
        ("analyze_law", "--bootstrap"),
        ("check_graph_seeds", "--n-boot"),
        ("check_threshold_sensitivity", "--quick"),
        ("check_two_room_hetero", "--refit"),
        ("check_two_room_sparse", "--refit"),
        ("run_ui", "--allow-remote"),
    ],
)
def test_research_script_help_exposes_options(script, flag):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / f"{script}.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert flag in result.stdout


def test_analyze_law_missing_input_is_a_clear_cli_error(tmp_path, monkeypatch):
    module = _load_script("analyze_law")
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(sys, "argv", ["analyze_law.py", "--input", str(missing)])
    with pytest.raises(SystemExit, match="chạy scripts/run_all.py"):
        module.main()


def test_heavy_tail_missing_calibration_is_a_clear_cli_error(tmp_path, monkeypatch):
    module = _load_script("check_heavy_tail")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="run_all.py --only graph"):
        module.main()


@pytest.mark.parametrize(
    "script,argv,expected",
    [
        ("check_two_room_hetero", ["--quick", "--refit"], "two_room_hetero_quick.csv not found"),
        ("check_two_room_replicates", ["--quick", "--refit"], "two_room_replicates_quick.csv not found"),
        ("check_two_room_sparse", ["--quick", "--refit"], "two_room_sparse_quick.csv not found"),
    ],
)
def test_refit_scripts_reject_missing_raw_csv_with_actionable_message(
    script, argv, expected, tmp_path, monkeypatch
):
    module = _load_script(script)
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", [f"{script}.py", *argv])
    with pytest.raises(SystemExit, match=expected.replace(".", r"\.")):
        module.main()


def test_ui_requires_explicit_opt_in_for_remote_bind(monkeypatch):
    from hive import webapp

    monkeypatch.setattr(sys, "argv", ["run_ui.py", "--host", "0.0.0.0", "--no-browser"])
    with pytest.raises(SystemExit) as exc:
        webapp.main()
    assert exc.value.code == 2


def test_project_license_is_declared_in_package_metadata():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert 'license = {file = "LICENSE"}' in pyproject
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Tienledaithandong" in license_text
