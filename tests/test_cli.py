import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hive import provenance

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


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "scripts" / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(module, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_ALL", {
        mode: {"reproduction": {"N": 30, "steps": 8, "seeds": 3},
               "graph": {"N": 30, "steps": 12, "seeds": 3, "ws_p": [0.1], "rgg_c": [2.0]}}
        for mode in ("quick", "full")
    })

    def plot(*args):
        directory = args[-1]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "test.png"
        path.write_bytes(b"plot")
        return [path]

    for name in ("plot_reproduction", "plot_law", "plot_trajectories"):
        monkeypatch.setattr(module, name, plot)
    return module


def test_quick_run_preserves_all_full_artifacts(runner, monkeypatch):
    runner.RESULTS.mkdir()
    runner.FIGURES.mkdir()
    sentinels = []
    for directory, names in (
        (runner.RESULTS, ("reproduction.npz", "graph_sweep.csv", "graph_curves.npz", "config_summary.csv", "reproduction_metadata.json", "graph_sweep_metadata.json")),
        (runner.FIGURES, ("test.png",)),
    ):
        for name in names:
            path = directory / name
            path.write_bytes(b"full sentinel")
            sentinels.append(path)
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--quick"])
    runner.main()
    assert all(path.read_bytes() == b"full sentinel" for path in sentinels)
    quick = runner.RESULTS / "quick"
    for name in ("reproduction.npz", "graph_sweep.csv", "graph_curves.npz", "config_summary.csv"):
        assert (quick / name).is_file()
    for name in ("reproduction_metadata.json", "graph_sweep_metadata.json"):
        meta = json.loads((quick / name).read_text(encoding="utf-8"))
        assert meta["mode"] == "quick"
        assert meta["config"]["seeds"] == 3
    assert (runner.FIGURES / "quick" / "test.png").is_file()


def test_replot_preserves_simulation_data_and_metadata(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--quick"])
    runner.main()
    quick = runner.RESULTS / "quick"
    original = {path: path.read_bytes() for path in quick.iterdir()}
    monkeypatch.setattr(runner, "run_graph_sweep", lambda **kwargs: pytest.fail("replot ran a simulation"))
    monkeypatch.setattr(runner, "run_reproduction", lambda **kwargs: pytest.fail("replot ran a simulation"))
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--quick", "--replot"])
    runner.main()
    assert all(path.read_bytes() == data for path, data in original.items())


def test_partial_run_does_not_relabel_other_experiment(runner, monkeypatch):
    runner.RESULTS.mkdir()
    path = runner.RESULTS / "graph_sweep_metadata.json"
    path.write_bytes(b"original graph provenance")
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--only", "reproduction"])
    runner.main()
    assert path.read_bytes() == b"original graph provenance"
    assert (runner.RESULTS / "reproduction_metadata.json").is_file()


@pytest.mark.parametrize("text,font", [("seed=0", "Mono"), ("cầu/bậc", None), ("t₉₀", "Mono")])
def test_report_code_spans_have_glyphs(text, font):
    spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts" / "build_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register_fonts()
    assert f'name="{font or module.FAMILY}"' in module.inline(f"`{text}`")


def test_replicate_refit_hashes_both_saved_inputs_without_relabeling_summary(tmp_path, monkeypatch):
    import pandas as pd

    spec = importlib.util.spec_from_file_location("check_two_room_replicates", ROOT / "scripts" / "check_two_room_replicates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(module, "compute_fits", lambda summary, df: {})
    raw = tmp_path / "two_room_replicates_quick.csv"
    summary = tmp_path / "two_room_replicates_summary_quick.csv"
    original = tmp_path / "two_room_replicates_metadata_quick.json"
    pd.DataFrame({"seed": [0]}).to_csv(raw, index=False)
    pd.DataFrame(columns=["N", "bridge", "phi", "t_first_median", "t_first_hit_rate", "t50_median", "t50_hit_rate", "t90_median", "t90_hit_rate", "arrivals_mean", "extinctions_mean"]).to_csv(summary, index=False)
    original.write_bytes(b"historical simulation metadata")
    before = {path: path.read_bytes() for path in (raw, summary, original)}
    monkeypatch.setattr(sys, "argv", ["check_two_room_replicates.py", "--quick", "--refit"])
    module.main()
    assert all(path.read_bytes() == data for path, data in before.items())
    meta = json.loads((tmp_path / "two_room_replicates_fits_metadata_quick.json").read_text(encoding="utf-8"))
    assert set(meta["inputs"]) == {raw.name, summary.name}
    assert meta["config"]["operation"] == "refit"
    assert "summary_n_boot" not in meta["config"]


def test_report_builds_unicode_code_and_records_pdf_inputs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts" / "build_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "report.md"
    source.write_text("# Báo cáo\n\n`cầu/bậc`\n\n```\nt₉₀ = 1 + L\n```\n", encoding="utf-8")
    target = tmp_path / "report.pdf"
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "TARGET", target)
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    assert module.build() == target
    assert target.read_bytes().startswith(b"%PDF")
    meta = json.loads((tmp_path / "report_metadata.json").read_text(encoding="utf-8"))
    assert meta["inputs"][source.name] == provenance.sha256_file(source)
    assert meta["artifacts"][target.name] == provenance.sha256_file(target)
