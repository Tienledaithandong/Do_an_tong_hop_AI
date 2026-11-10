import hashlib
import json

from hive import provenance


def test_metadata_records_config_versions_seeds_and_artifact_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "_git", lambda *args: "abc123" if args[0] == "rev-parse" else "")
    monkeypatch.setattr(provenance, "versions", lambda: {"python": "3.13", "numpy": "2.5.2"})
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    data = tmp_path / "data.csv"
    data.write_bytes(b"seed,t90\n0,10\n")
    path = provenance.write_metadata(
        tmp_path / "results", "quick", {"seeds": 3},
        name="example_metadata", inputs=[source], artifacts=[data],
        extra={"swarm_seed_start": 0, "swarm_seed_stop_exclusive": 3},
    )
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["mode"] == "quick"
    assert meta["config"] == {"seeds": 3}
    assert meta["git_sha"] == "abc123"
    assert meta["git_dirty"] is False
    assert meta["created_utc"].endswith("+00:00")
    assert meta["versions"]["numpy"] == "2.5.2"
    assert meta["extra"]["swarm_seed_stop_exclusive"] == 3
    assert meta["inputs"]["src/example.py"] == provenance.sha256_file(source)
    assert meta["artifacts"]["data.csv"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert meta["source_sha256"]["src/example.py"] == provenance.sha256_file(source)


def test_metadata_distinguishes_dirty_and_unavailable_git(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    meta = provenance.run_metadata("full", {})
    assert meta["git_sha"] is None
    assert meta["git_dirty"] is None
    monkeypatch.setattr(provenance, "_git", lambda *args: "?? example.py")
    assert provenance.run_metadata("full", {})["git_dirty"] is True


def test_source_hash_changes_even_before_a_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    source = tmp_path / "scripts" / "experiment.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    before = provenance.run_metadata("full", {})["source_sha256"]
    source.write_text("value = 2\n", encoding="utf-8")
    after = provenance.run_metadata("full", {})["source_sha256"]
    assert before["scripts/experiment.py"] != after["scripts/experiment.py"]


def test_relative_input_keeps_its_workspace_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    path = tmp_path / "results" / "quick" / "graph_sweep.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"data")
    assert provenance.file_records([path.relative_to(tmp_path)]) == {
        "results/quick/graph_sweep.csv": provenance.sha256_file(path)
    }
