"""Bind aggregate reports to their exact forecast artifacts and frozen protocol."""

from __future__ import annotations

import json
from pathlib import Path

from .funding_labels import sha256


def verify_report_inputs(output: Path, spec: dict) -> dict:
    """Reject changed sources/selection; write hashes before scoring forecasts."""
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["protocol"] != spec:
        raise ValueError("Report config differs from executed forecast protocol")
    selection = json.loads((output / "selection.json").read_text())
    if selection["identity"]["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("Selection and forecast manifest disagree")
    root = Path(__file__).resolve().parents[2]
    identity = selection["identity"]
    if sha256(root / spec["ticker_path"]) != identity["ticker_sha256"]:
        raise ValueError("Ticker source changed since selection")
    for name, digest in identity["code_sha256"].items():
        if sha256(root / f"v2/src/{name}.py") != digest:
            raise ValueError(f"Forecast code changed since selection: {name}")
    record = {
        "prediction_sha256": sha256(output / "predictions.csv.gz"),
        "folds_sha256": sha256(output / "folds.csv"),
        "selection_sha256": sha256(output / "selection.json"),
        "config_sha256": manifest["config_sha256"],
        "reporting_code_sha256": {
            name: sha256(root / f"v2/src/{name}.py")
            for name in ["metrics", "reporting", "provenance"]
        },
    }
    path = output / "forecast_artifact_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ["prediction_sha256", "folds_sha256", "selection_sha256", "config_sha256"]:
            if previous[key] != record[key]:
                raise ValueError(
                    "Previously reported forecast artifacts changed; use a new run directory"
                )
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record
