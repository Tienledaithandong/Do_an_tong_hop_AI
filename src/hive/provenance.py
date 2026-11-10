"""Configuration and code provenance for newly generated artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
_VERSIONED = ("numpy", "scipy", "pandas", "matplotlib", "networkx", "reportlab", "Pillow")


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(paths: Iterable[Path]) -> dict[str, str]:
    records = {}
    for path in sorted(paths):
        path = path.resolve()
        name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        records[name] = sha256_file(path)
    return records


def versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in _VERSIONED:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def run_metadata(mode: str, config: dict, extra: dict | None = None) -> dict:
    dirty = _git("status", "--porcelain")
    sources = [
        path for directory in (ROOT / "src", ROOT / "scripts")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".js", ".css", ".html"}
    ]
    sources += [path for path in (ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "requirements-lock.txt") if path.is_file()]
    meta = {
        "schema": 1,
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrypoint": Path(sys.argv[0]).name,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "versions": versions(),
        "source_sha256": file_records(sources),
        "config": config,
    }
    if extra:
        meta["extra"] = extra
    return meta


def write_metadata(
    results_dir: Path,
    mode: str,
    config: dict,
    suffix: str = "",
    extra: dict | None = None,
    name: str = "run_metadata",
    inputs: Iterable[Path] = (),
    artifacts: Iterable[Path] = (),
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(mode, config, extra)
    meta["inputs"] = file_records(inputs)
    meta["artifacts"] = file_records(artifacts)
    path = results_dir / f"{name}{suffix}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
