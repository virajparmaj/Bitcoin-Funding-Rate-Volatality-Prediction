"""Immutable stage transactions, with an atomic directory as the commit record."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .funding_labels import sha256

STAGES = (
    "validate-data",
    "baselines",
    "models",
    "sensitivities",
    "replay",
    "report",
    "reproduction",
)


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def execution_identity(manifest: dict) -> dict:
    """Git packaging and timestamps are metadata, not numerical dependencies."""
    return {
        key: manifest[key]
        for key in (
            "config_sha256",
            "protocol",
            "archives",
            "ticker_sha256",
            "code_sha256",
            "python",
            "packages",
        )
    }


def check_store(output: Path, identity: dict) -> dict[str, Path]:
    """Verify every completed stage and public artifact before any writes."""
    stages = output / "_stages"
    if (output / "manifest.json").exists() and not stages.exists():
        raise ValueError("Legacy evidence is immutable; use a new output directory")
    artifacts = {}
    for name in STAGES:
        directory = stages / name
        if not directory.exists():
            continue
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["identity_sha256"] != digest(identity):
            raise ValueError("Execution identity changed; use a new output directory")
        dependencies = {k: sha256(v) for k, v in artifacts.items()}
        if receipt["inputs"] != dependencies:
            raise ValueError(f"Stage dependencies changed: {name}")
        for filename, expected in receipt["outputs"].items():
            path = directory / filename
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Corrupted stage artifact: {path}")
            public = output / filename
            if (public.exists() or public.is_symlink()) and (
                not public.is_file() or sha256(public) != expected
            ):
                raise ValueError(f"Corrupted public artifact: {public}")
            if filename in artifacts:
                raise ValueError(f"Stage attempted to replace evidence: {filename}")
            artifacts[filename] = path
    return artifacts


@contextmanager
def run_lock(output: Path):
    """Serialize writers without holding a lock inside immutable evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is writing this run") from error
        yield


def publish_links(output: Path, artifacts: dict[str, Path]) -> None:
    """Repair interrupted publication only after canonical evidence verifies."""
    for name, path in artifacts.items():
        public = output / name
        if not public.exists():
            public.symlink_to(path.relative_to(output))


def execute_stage(output: Path, stage: str, identity: dict, action, recheck) -> None:
    """Commit new artifacts together; failures cannot overwrite completed data."""
    with run_lock(output):
        artifacts = check_store(output, identity)
        destination = output / "_stages" / stage
        if destination.exists():
            publish_links(output, artifacts)
            print(f"Verified completed stage: {stage}", flush=True)
            return
        position = STAGES.index(stage)
        if position and not (output / "_stages" / STAGES[position - 1]).exists():
            raise ValueError(f"Run {STAGES[position - 1]} before {stage}")
        before = {k: sha256(v) for k, v in artifacts.items()}
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}-{stage}-", dir=output.parent
        ) as tmp:
            work = Path(tmp)
            for name, path in artifacts.items():
                shutil.copyfile(path, work / name)
            action(work)
            if any(not (work / k).is_file() or sha256(work / k) != v for k, v in before.items()):
                raise ValueError("Stage attempted to overwrite prior evidence")
            additions = {p.name: sha256(p) for p in work.iterdir() if p.name not in before}
            if any((output / name).exists() or (output / name).is_symlink() for name in additions):
                raise ValueError("Unregistered output already exists; use a new output directory")
            if not additions:
                raise ValueError("Stage produced no evidence")
            if recheck() != identity:
                raise ValueError("Inputs or implementation changed during execution")
            check_store(output, identity)
            for name in before:
                (work / name).unlink()
            receipt = dict(
                stage=stage, identity_sha256=digest(identity), inputs=before, outputs=additions
            )
            (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(work, destination)
        publish_links(output, check_store(output, identity))
        print(f"Committed stage: {stage}", flush=True)
