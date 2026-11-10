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


def provenance_coverage(payload: dict[str, bytes]) -> dict:
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
        if name.startswith(("results/", "figures/")) and "metadata" not in name
    }
    return {"attributed_artifacts": sorted(artifacts & matching),
            "unattributed_artifacts": sorted(artifacts - matching),
            "stale_metadata": sorted(stale),
            "dependency_drift": dependencies}


def build_package(root: Path, output: Path, mode: str) -> dict:
    if mode == "full" and not (root / "report" / "report.pdf").is_file():
        raise ValueError("Build report/report.pdf before creating a full defense package")
    payload = {path.relative_to(root).as_posix(): path.read_bytes() for path in collect_files(root, mode)}
    if not any(name.startswith("results/") and "metadata" not in name for name in payload):
        raise ValueError(f"No {mode} results found; run the experiments first")
    manifest = {
        "schema": 1, "mode": mode,
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


def verify_package(path: Path) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") != 1 or manifest.get("mode") not in {"full", "quick"}:
            raise ValueError("Unsupported manifest")
        if set(names) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Archive inventory differs from manifest")
        payload = {}
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {name}")
            payload[name] = data
        if provenance_coverage(payload) != manifest["provenance_coverage"]:
            raise ValueError("Provenance coverage differs from manifest")
    checksum = path.with_suffix(path.suffix + ".sha256")
    if checksum.is_file() and checksum.read_text(encoding="utf-8").split()[0] != sha256_file(path):
        raise ValueError("Archive checksum differs from SHA256 sidecar")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="bundle smoke-test artifacts instead of full results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="verify an existing ZIP without extracting it")
    args = parser.parse_args()
    try:
        if args.verify:
            manifest = verify_package(args.verify)
            print(f"Verified {len(manifest['files'])} files ({manifest['mode']}) in {args.verify}")
        else:
            mode = "quick" if args.quick else "full"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = args.output or ROOT / "out" / f"defense-{mode}-{stamp}.zip"
            manifest = build_package(ROOT, output, mode)
            verify_package(output)
            print(f"Wrote {output}: {len(manifest['files'])} files; SHA256 sidecar verified")
        coverage = manifest["provenance_coverage"]
        missing = coverage["unattributed_artifacts"]
        if missing:
            print(f"Warning: {len(missing)} artifacts lack matching run metadata; listed in manifest, not retroactively attributed")
        if coverage["dependency_drift"]:
            print(f"Warning: {len(coverage['dependency_drift'])} metadata records have missing or changed inputs/source; listed in manifest")
    except (OSError, ValueError, BadZipFile, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Artifact package failed: {exc}\n")


if __name__ == "__main__":
    main()
