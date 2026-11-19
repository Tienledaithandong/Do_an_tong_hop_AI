"""Bundle full or quick artifacts and verify their SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hive.provenance import run_metadata, sha256_file

RESULT_PREFIXES = (
    "reproduction", "graph_sweep", "graph_curves", "config_summary", "graph_seeds",
    "heavy_tail", "law_diagnostics", "two_room_", "threshold_sensitivity",
)

# Files needed for the reported experiment suite, rather than merely one
# arbitrary result plus an existing PDF. Keep in sync with README's figure map.
FULL_RESULTS = (
    "reproduction.npz", "graph_sweep.csv", "graph_curves.npz", "config_summary.csv",
    "heavy_tail.csv", "law_diagnostics.json", "two_room_source.csv",
    "two_room_source_summary.csv", "two_room_replicates.csv",
    "two_room_replicates_summary.csv", "two_room_replicates_fits.json",
    "two_room_sparse.csv", "two_room_sparse_summary.csv", "two_room_sparse_fits.json",
    "two_room_hetero.csv", "two_room_hetero_summary.csv", "two_room_hetero_fits.json",
    "graph_seeds.csv", "graph_seeds_summary.csv", "graph_seeds_t90.npz",
    "graph_seeds_law.json", "threshold_sensitivity_law.csv",
    "threshold_sensitivity_two_room.csv", "threshold_sensitivity_fit.json",
)
FIGURE_NAMES = (
    "fig1_reproduction", "fig2_law", "fig3_trajectories", "fig4_two_room_source",
    "fig5_two_room_replicates", "fig6_two_room_sparse", "fig7_two_room_hetero",
    "fig8_graph_seeds", "fig9_threshold_sensitivity",
)


def required_files(mode: str) -> set[str]:
    if mode == "full":
        results = {f"results/{name}" for name in FULL_RESULTS}
        figures = {f"figures/{name}.{ext}" for name in FIGURE_NAMES for ext in ("png", "pdf")}
        return results | figures | {"report/report.md", "report/report.pdf"}
    results = {f"results/quick/{name}" for name in FULL_RESULTS[:4]}
    figures = {f"figures/quick/{name}.{ext}" for name in FIGURE_NAMES[:3] for ext in ("png", "pdf")}
    return results | figures


def collect_files(root: Path, mode: str) -> list[Path]:
    files = [
        root / name for name in ("README.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt", "LICENSE", ".gitignore")
        if (root / name).is_file()
    ]
    for directory in ("src", "scripts", "tests", ".github/workflows"):
        files.extend(
            path for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".html", ".css", ".js", ".yml", ".yaml"}
        )
    for directory in ("results", "figures"):
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix not in {".csv", ".json", ".npz", ".png", ".pdf"}:
                continue
            if directory == "results" and not path.name.startswith(RESULT_PREFIXES):
                continue
            if directory == "figures" and not path.name.startswith("fig"):
                continue
            is_quick = "quick" in path.relative_to(root / directory).parts or path.stem.endswith("_quick")
            if is_quick == (mode == "quick"):
                files.append(path)
    files.append(root / "report" / "report.md")
    if mode == "full":
        files.extend(root / "report" / name for name in ("report.pdf", "report_metadata.json") if (root / "report" / name).is_file())
    return sorted(set(path for path in files if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())))


def provenance_coverage(payload: dict[str, bytes], *, include_report: bool = True) -> dict:
    matching = set()
    stale = set()
    dependencies = {}
    for name, data in payload.items():
        if "metadata" not in name or not name.endswith(".json"):
            continue
        meta = json.loads(data)
        for artifact, expected in meta.get("artifacts", {}).items():
            if artifact in payload:
                if hashlib.sha256(payload[artifact]).hexdigest() == expected:
                    matching.add(artifact)
                else:
                    stale.add(name)
        for field in ("inputs", "source_sha256"):
            mismatched = []
            missing = []
            for dependency, expected in meta.get(field, {}).items():
                if dependency not in payload:
                    missing.append(dependency)
                elif hashlib.sha256(payload[dependency]).hexdigest() != expected:
                    mismatched.append(dependency)
            if mismatched or missing:
                dependencies.setdefault(name, {})[field] = {"mismatched": sorted(mismatched), "missing": sorted(missing)}
    artifacts = {
        name for name in payload
        if (name.startswith(("results/", "figures/")) and "metadata" not in name)
        or (include_report and name == "report/report.pdf")
    }
    return {"attributed_artifacts": sorted(artifacts & matching),
            "unattributed_artifacts": sorted(artifacts - matching),
            "stale_metadata": sorted(stale),
            "dependency_drift": dependencies}


def strict_issues(payload: dict[str, bytes], mode: str) -> list[str]:
    """Check completeness and current provenance, allowing superseded renders.

    A replot may make an older simulation sidecar's figure hashes stale, while
    its data remains current. Each present artifact instead needs at least one
    matching sidecar whose own recorded inputs and sources match the snapshot.
    This does not establish numerical or scientific correctness.
    """
    coverage = provenance_coverage(payload)
    missing = sorted(required_files(mode) - payload.keys())
    issues = [f"missing required file: {name}" for name in missing]
    issues += [f"no matching metadata: {name}" for name in coverage["unattributed_artifacts"]]

    matched = set(coverage["attributed_artifacts"])
    clean = set()
    blocked: dict[str, list[str]] = {}
    for name, data in payload.items():
        if "metadata" not in name or not name.endswith(".json"):
            continue
        meta = json.loads(data)
        current = {
            artifact for artifact, expected in meta.get("artifacts", {}).items()
            if artifact in matched and hashlib.sha256(payload[artifact]).hexdigest() == expected
        }
        if name in coverage["dependency_drift"]:
            for artifact in current:
                blocked.setdefault(artifact, []).append(name)
        else:
            clean.update(current)
    for artifact in sorted(matched - clean):
        records = ", ".join(sorted(blocked.get(artifact, [])))
        issues.append(f"metadata has missing or changed inputs/source for {artifact}: {records}")
    return issues


def require_strict(payload: dict[str, bytes], mode: str) -> None:
    issues = strict_issues(payload, mode)
    if issues:
        raise ValueError("Strict provenance/completeness check failed:\n  " + "\n  ".join(issues))


def build_package(root: Path, output: Path, mode: str, *, strict: bool = False) -> dict:
    if mode == "full" and not (root / "report" / "report.pdf").is_file():
        raise ValueError("Build report/report.pdf before creating a full defense package")
    payload = {path.relative_to(root).as_posix(): path.read_bytes() for path in collect_files(root, mode)}
    if not any(name.startswith("results/") and "metadata" not in name for name in payload):
        raise ValueError(f"No {mode} results found; run the experiments first")
    if strict:
        require_strict(payload, mode)
    manifest = {
        "schema": 2, "mode": mode,
        "packaged_at": run_metadata(mode, {}, {"operation": "package_snapshot"}),
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in payload.items()},
        "provenance_coverage": provenance_coverage(payload),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    return manifest


def verify_package(path: Path, *, strict: bool = False) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") not in {1, 2} or manifest.get("mode") not in {"full", "quick"}:
            raise ValueError("Unsupported manifest")
        if set(names) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Archive inventory differs from manifest")
        payload = {}
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {name}")
            payload[name] = data
        if provenance_coverage(payload, include_report=manifest["schema"] >= 2) != manifest["provenance_coverage"]:
            raise ValueError("Provenance coverage differs from manifest")
        if strict:
            require_strict(payload, manifest["mode"])
    checksum = path.with_suffix(path.suffix + ".sha256")
    if checksum.is_file() and checksum.read_text(encoding="utf-8").split()[0] != sha256_file(path):
        raise ValueError("Archive checksum differs from SHA256 sidecar")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="bundle smoke-test artifacts instead of full results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="verify an existing ZIP without extracting it")
    parser.add_argument("--strict", action="store_true", help="require the complete artifact set and matching current provenance")
    args = parser.parse_args()
    try:
        if args.verify:
            manifest = verify_package(args.verify, strict=args.strict)
            print(f"Verified {len(manifest['files'])} files ({manifest['mode']}) in {args.verify}")
        else:
            mode = "quick" if args.quick else "full"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = args.output or ROOT / "out" / f"defense-{mode}-{stamp}.zip"
            manifest = build_package(ROOT, output, mode, strict=args.strict)
            verify_package(output, strict=args.strict)
            print(f"Wrote {output}: {len(manifest['files'])} files; SHA256 sidecar verified")
        coverage = manifest["provenance_coverage"]
        missing = coverage["unattributed_artifacts"]
        if missing:
            print(f"Warning: {len(missing)} artifacts lack matching run metadata; listed in manifest, not retroactively attributed")
        if coverage["dependency_drift"]:
            print(f"Warning: {len(coverage['dependency_drift'])} metadata records have missing or changed inputs/source; listed in manifest")
        if coverage["stale_metadata"]:
            print(f"Warning: {len(coverage['stale_metadata'])} metadata records have stale artifact hashes; listed in manifest")
    except (OSError, ValueError, BadZipFile, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Artifact package failed: {exc}\n")


if __name__ == "__main__":
    main()
