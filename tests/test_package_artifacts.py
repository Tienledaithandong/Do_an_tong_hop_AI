import importlib.util
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from hive import provenance

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_artifacts", ROOT / "scripts" / "package_artifacts.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    for directory in ("results/quick", "figures/quick", "report", "src/hive", "out"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for name, data in {
        "README.md": b"project", "report/report.md": b"report", "report/report.pdf": b"pdf",
        "src/hive/model.py": b"model", "results/graph_sweep.csv": b"full",
        "results/quick/graph_sweep.csv": b"quick", "results/graph_seeds_quick.csv": b"quick calibration",
        "results/sample_config.csv": b"scratch", "figures/fig1.png": b"full figure",
        "figures/quick/fig1.png": b"quick figure", "figures/fig8_graph_seeds_quick.png": b"quick calibration figure",
        ".env": b"secret", "out/unrelated.zip": b"old zip",
    }.items():
        (tmp_path / name).write_bytes(data)
    provenance.write_metadata(
        tmp_path / "results", "full", {"seeds": 100}, name="graph_sweep_metadata",
        artifacts=[tmp_path / "results/graph_sweep.csv"],
    )
    return tmp_path


def test_full_package_excludes_quick_secrets_and_scratch_and_verifies(artifacts):
    output = artifacts / "out/full.zip"
    manifest = package.build_package(artifacts, output, "full")
    assert package.verify_package(output) == manifest
    names = set(manifest["files"])
    assert "results/graph_sweep.csv" in names
    assert "report/report.pdf" in names
    assert "src/hive/model.py" in names
    assert not any("quick" in name for name in names)
    assert ".env" not in names
    assert "results/sample_config.csv" not in names
    assert not any(name.startswith("out/") for name in names)
    coverage = manifest["provenance_coverage"]
    assert "results/graph_sweep.csv" in coverage["attributed_artifacts"]
    assert "figures/fig1.png" in coverage["unattributed_artifacts"]
    assert output.with_suffix(".zip.sha256").is_file()


def test_quick_package_excludes_full_results_and_full_report(artifacts):
    manifest = package.build_package(artifacts, artifacts / "out/quick.zip", "quick")
    names = set(manifest["files"])
    assert "results/quick/graph_sweep.csv" in names
    assert "results/graph_seeds_quick.csv" in names
    assert "figures/fig8_graph_seeds_quick.png" in names
    assert "results/graph_sweep.csv" not in names
    assert "figures/fig1.png" not in names
    assert "report/report.pdf" not in names


def test_package_refuses_to_overwrite_existing_archive(artifacts):
    output = artifacts / "out/full.zip"
    package.build_package(artifacts, output, "full")
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        package.build_package(artifacts, output, "full")
    assert output.read_bytes() == before


def test_verify_detects_modified_payload(artifacts):
    output = artifacts / "out/full.zip"
    package.build_package(artifacts, output, "full")
    with ZipFile(output) as original:
        payload = {name: original.read(name) for name in original.namelist()}
    payload["results/graph_sweep.csv"] = b"tampered"
    changed = artifacts / "out/tampered.zip"
    with ZipFile(changed, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        package.verify_package(changed)


def test_stale_metadata_is_reported_not_silently_attributed(artifacts):
    (artifacts / "results/graph_sweep.csv").write_bytes(b"changed since run")
    manifest = package.build_package(artifacts, artifacts / "out/stale.zip", "full")
    coverage = manifest["provenance_coverage"]
    assert "results/graph_sweep.csv" in coverage["unattributed_artifacts"]
    assert "results/graph_sweep_metadata.json" in coverage["stale_metadata"]


def test_archive_checksum_sidecar_is_checked(artifacts):
    output = artifacts / "out/full.zip"
    package.build_package(artifacts, output, "full")
    output.with_suffix(".zip.sha256").write_text("0" * 64 + "  full.zip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 sidecar"):
        package.verify_package(output)


def test_inventory_and_provenance_coverage_cannot_be_faked(artifacts):
    output = artifacts / "out/full.zip"
    package.build_package(artifacts, output, "full")
    with ZipFile(output) as original:
        payload = {name: original.read(name) for name in original.namelist()}
    manifest = json.loads(payload["manifest.json"])
    manifest["provenance_coverage"]["unattributed_artifacts"] = []
    payload["manifest.json"] = json.dumps(manifest).encode("utf-8")
    changed = artifacts / "out/faked.zip"
    with ZipFile(changed, "w") as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="Provenance coverage"):
        package.verify_package(changed)


def test_changed_inputs_and_source_are_reported_separately_from_artifact_hashes(artifacts):
    fitted = artifacts / "results" / "graph_seeds_law.json"
    fitted.write_bytes(b"fitted")
    source = artifacts / "src" / "hive" / "model.py"
    raw = artifacts / "results" / "graph_sweep.csv"
    provenance.write_metadata(
        artifacts / "results", "full", {}, name="graph_seeds_fits_metadata",
        inputs=[raw], artifacts=[fitted],
    )
    raw.write_bytes(b"changed input")
    source.write_bytes(b"changed source")
    manifest = package.build_package(artifacts, artifacts / "out/drift.zip", "full")
    coverage = manifest["provenance_coverage"]
    assert "results/graph_seeds_law.json" in coverage["attributed_artifacts"]
    drift = coverage["dependency_drift"]["results/graph_seeds_fits_metadata.json"]
    assert drift["inputs"]["mismatched"] == ["results/graph_sweep.csv"]
    assert drift["source_sha256"]["mismatched"] == ["src/hive/model.py"]
    assert package.verify_package(artifacts / "out/drift.zip") == manifest


def test_missing_provenance_input_is_reported(artifacts):
    external = artifacts / "out" / "not-bundled.csv"
    external.write_bytes(b"external input")
    provenance.write_metadata(
        artifacts / "results", "full", {}, name="graph_sweep_metadata",
        inputs=[external], artifacts=[artifacts / "results" / "graph_sweep.csv"],
    )
    manifest = package.build_package(artifacts, artifacts / "out/missing-input.zip", "full")
    drift = manifest["provenance_coverage"]["dependency_drift"]["results/graph_sweep_metadata.json"]
    assert drift["inputs"] == {"mismatched": [], "missing": ["out/not-bundled.csv"]}


def test_full_package_requires_report_and_results(tmp_path):
    with pytest.raises(ValueError, match="report/report.pdf"):
        package.build_package(tmp_path, tmp_path / "out/missing.zip", "full")


@pytest.fixture
def complete_artifacts(artifacts):
    """Small synthetic complete inventory; hashes and lineage are real."""
    for name in package.required_files("full"):
        path = artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(name.encode())
    provenance.write_metadata(
        artifacts / "results", "full", {}, name="graph_sweep_metadata",
        artifacts=[artifacts / name for name in package.required_files("full")
                   if name.startswith(("results/", "figures/"))],
    )
    provenance.write_metadata(
        artifacts / "report", "full", {}, name="report_metadata",
        inputs=[artifacts / "report/report.md", artifacts / "figures/fig1_reproduction.png"],
        artifacts=[artifacts / "report/report.pdf"],
    )
    # The fixture's extra figure is not part of the mandatory inventory but is
    # included in the package and must also have provenance in strict mode.
    provenance.write_metadata(
        artifacts / "results", "full", {}, name="graph_seeds_metadata",
        artifacts=[artifacts / "figures/fig1.png"],
    )
    return artifacts


def test_report_pdf_is_in_coverage_and_without_metadata_is_unattributed(artifacts):
    manifest = package.build_package(artifacts, artifacts / "out/snapshot.zip", "full")
    assert "report/report.pdf" in manifest["provenance_coverage"]["unattributed_artifacts"]


def test_strict_refuses_incomplete_before_creating_zip(artifacts):
    output = artifacts / "out/incomplete.zip"
    with pytest.raises(ValueError, match="missing required file: results/reproduction.npz"):
        package.build_package(artifacts, output, "full", strict=True)
    assert not output.exists()


def test_complete_package_can_be_built_and_verified_strictly(complete_artifacts):
    root = complete_artifacts
    output = root / "out/complete.zip"
    manifest = package.build_package(root, output, "full", strict=True)
    assert "report/report.pdf" in manifest["provenance_coverage"]["attributed_artifacts"]
    assert package.verify_package(output, strict=True) == manifest


def test_strict_rejects_changed_report_input_even_if_pdf_hash_matches(complete_artifacts):
    root = complete_artifacts
    (root / "report/report.md").write_bytes(b"changed report")
    output = root / "out/stale-report.zip"
    manifest = package.build_package(root, output, "full")
    assert "report/report.pdf" in manifest["provenance_coverage"]["attributed_artifacts"]
    with pytest.raises(ValueError, match="missing or changed inputs/source for report/report.pdf"):
        package.verify_package(output, strict=True)


def test_strict_accepts_superseded_render_when_new_provenance_is_valid(complete_artifacts):
    root = complete_artifacts
    image = root / "figures/fig1_reproduction.png"
    image.write_bytes(b"replotted image")
    provenance.write_metadata(
        root / "results", "full", {}, name="reproduction_render_metadata",
        inputs=[root / "results/reproduction.npz"], artifacts=[image],
    )
    # Regenerate report after replotting, as its image input changed.
    provenance.write_metadata(
        root / "report", "full", {}, name="report_metadata",
        inputs=[root / "report/report.md", image], artifacts=[root / "report/report.pdf"],
    )
    manifest = package.build_package(root, root / "out/replotted.zip", "full", strict=True)
    assert "results/graph_sweep_metadata.json" in manifest["provenance_coverage"]["stale_metadata"]


def test_strict_rejects_mismatched_source_when_no_current_lineage(complete_artifacts):
    root = complete_artifacts
    (root / "src/hive/model.py").write_bytes(b"changed code")
    with pytest.raises(ValueError, match="missing or changed inputs/source"):
        package.build_package(root, root / "out/drift.zip", "full", strict=True)
