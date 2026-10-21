"""Bind reports to their complete source, forecast and verification evidence."""

import json
from pathlib import Path

from .funding_labels import sha256
from .study_store import digest, execution_identity


def verify_report_inputs(output: Path, spec: dict) -> dict:
    """Verify dependencies before the transactional report stage publishes anything."""
    manifest = json.loads((output / "manifest.json").read_text())
    identity = execution_identity(manifest)
    if manifest["protocol"] != spec:
        raise ValueError("Report config differs from executed forecast protocol")
    selection = json.loads((output / "selection.json").read_text())
    if selection["identity"] != identity:
        raise ValueError("Selection and forecast manifest disagree")
    for name in ["publication_delay_sensitivity.json", "forecast_replay.json"]:
        record = json.loads((output / name).read_text())
        if record["identity_sha256"] != digest(identity):
            raise ValueError(f"Stale verification evidence: {name}")
    checks = json.loads((output / "publication_delay_sensitivity.json").read_text())["checks"]
    if {r["label_delay_minutes"] for r in checks} != {0, 5, 15} or not all(
        r["same_origin_keys"] and r["identical_settled_features"] and r["identical_fold_membership"]
        for r in checks
    ):
        raise ValueError(
            "Publication assumptions change predictions; execute versioned sensitivity runs"
        )
    if json.loads((output / "forecast_replay.json").read_text())["status"] != "passed":
        raise ValueError("Forecast replay did not pass")
    record = dict(
        identity_sha256=digest(identity),
        artifacts={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    (output / "forecast_artifact_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
