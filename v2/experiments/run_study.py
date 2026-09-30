"""Run the frozen retrospective settlement study from the repository root."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from v2.src.evaluation import forecast, tune
from v2.src.features import add_settled_history, ticker_features
from v2.src.funding_labels import download_archives, load_labels, sha256
from v2.src.settlement import build_origin_panel
from v2.src.timebase import normalize_ticker
from v2.src.validate import validate_panel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "v2/configs/settlement_study.json"


def save_json(path: Path, value) -> None:
    """Write deterministic machine-readable evidence."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def configuration(path: Path):
    """Validate output boundaries and basic protocol domains before side effects."""
    spec = json.loads(path.read_text())
    for key, parent in (("output_dir", ROOT / "v2/results"), ("labels_dir", ROOT / "v2/data")):
        resolved = (ROOT / spec[key]).resolve()
        if not resolved.is_relative_to(parent.resolve()) or resolved == parent.resolve():
            raise ValueError(f"{key} must be a child of {parent}")
    if sorted(set(spec["horizons"])) != [1, 2, 4, 6]:
        raise ValueError("The declared study requires exactly horizons 1, 2, 4, 6")
    if spec["max_age_minutes"] < 0 or spec["label_delay_minutes"] < 0:
        raise ValueError("Negative availability tolerance/delay")
    if not (
        pd.Timestamp(spec["validation_start"])
        < pd.Timestamp(spec["test_start"])
        < pd.Timestamp(spec["test_end"])
    ):
        raise ValueError("Invalid chronological split")
    return spec


def inputs(spec):
    """Load immutable sources and require the entire declared month range."""
    directory = ROOT / spec["labels_dir"]
    paths = [
        directory / f"BTCUSDT-fundingRate-{month}.zip"
        for month in pd.period_range(spec["archive_start"], spec["archive_end"], freq="M")
    ]
    labels, manifest = load_labels(
        paths, spec["label_delay_minutes"], spec["label_match_tolerance_seconds"]
    )
    ticker, rejected = normalize_ticker(pd.read_csv(ROOT / spec["ticker_path"]))
    return ticker_features(ticker), labels, manifest, rejected


def panel_for(ticker, labels, spec, delay=0):
    """Create a causally valid cohort and explicitly remove incomplete warm-up."""
    panel, rejected = build_origin_panel(
        ticker, labels, spec["horizons"], spec["max_age_minutes"], delay
    )
    panel = add_settled_history(panel, labels, spec["ewma_spans"])
    missing = panel.settled_9.isna()
    warmup = panel.loc[missing, ["event_id", "horizon"]].assign(reason="settled_history_warmup")
    panel = panel.loc[~missing].reset_index(drop=True)
    validate_panel(panel, spec["max_age_minutes"])
    return panel, (
        pd.concat([x for x in [rejected, warmup] if len(x)], ignore_index=True)
        if len(rejected) + len(warmup)
        else rejected
    )


def manifest_for(spec, config_path, archives):
    """Capture source, configuration and implementation identity before evaluation."""
    code = list((ROOT / "v2/src").glob("*.py")) + list((ROOT / "v2/experiments").glob("*.py"))
    return dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha256(config_path),
        protocol=spec,
        archives=archives,
        ticker_sha256=sha256(ROOT / spec["ticker_path"]),
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(code)},
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        python=platform.python_version(),
        packages={
            p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "statsmodels"]
        },
        limitations=[
            "Retrospective, previously inspected historical sample",
            "Row-level arrival proxy; original field-level aggregation unverified",
            "Settlement publication delay is an assumption",
            "One venue/contract; no profitability claim",
        ],
    )


def validate_data(spec, output, ticker, labels, archives, rejected):
    """Write E0 coverage, exclusions, proxy discrepancies and admitted panel."""
    panel, excluded = panel_for(ticker, labels, spec)
    panel.to_csv(output / "panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    excluded.to_csv(output / "origin_exclusions.csv", index=False)
    rejected.to_csv(output / "ticker_exclusions.csv", index=False)
    events = panel.drop_duplicates("event_id")
    coverage = dict(
        ticker_rows=len(ticker),
        rejected_ticker_rows=len(rejected),
        raw_event_ids=ticker.event_id.nunique(),
        archive_labels=len(labels),
        admitted_events=events.event_id.nunique(),
        origin_rows=len(panel),
        origin_exclusions=len(excluded),
        archives=len(archives),
        label_offset_max_seconds=float(labels.offset_seconds.abs().max()),
        terminal_proxy_mae_bp=float((events.target - events.terminal_proxy).abs().mean() * 1e4),
    )
    save_json(output / "coverage.json", coverage)
    return panel


def frozen_choices(panel, spec, output, config_hash, source_hash):
    """Reuse tuning only when it matches exactly the current pre-test inputs."""
    path = output / "selection.json"
    critical = [
        "timebase",
        "funding_labels",
        "settlement",
        "features",
        "splits",
        "evaluation",
        "forecast_models",
    ]
    identity = dict(
        config_sha256=config_hash,
        ticker_sha256=source_hash,
        code_sha256={name: sha256(ROOT / f"v2/src/{name}.py") for name in critical},
        labels_sha256={
            p.name: sha256(p) for p in sorted((ROOT / spec["labels_dir"]).glob("*.zip"))
        },
    )
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Existing selection belongs to a different protocol/source; use a new output directory"
            )
        return saved["choices"]
    choices, ledger = tune(panel, spec)
    save_json(output / "validation_candidates.json", ledger)
    save_json(
        path,
        dict(
            identity=identity, selected_at=datetime.now(timezone.utc).isoformat(), choices=choices
        ),
    )
    return choices


def models(panel, ticker, labels, spec, output, choices):
    """Run main models, fixed-hyperparameter feature ablations and delayed sources."""
    results, fold_tables = [], []
    for group in spec["feature_groups"]:
        variant = "main" if group == "full" else group
        result, folds = forecast(panel, choices, spec, group, variant)
        results.append(result)
        fold_tables.append(folds)
    delayed, excluded = panel_for(ticker, labels, spec, spec["availability_delay_minutes"])
    excluded.to_csv(output / "delay_exclusions.csv", index=False)
    result, folds = forecast(delayed, choices, spec, "full", "delay_60m")
    results.append(result)
    fold_tables.append(folds)
    pd.concat(results, ignore_index=True).to_csv(
        output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    pd.concat(fold_tables, ignore_index=True).to_csv(output / "folds.csv", index=False)


def run(stage: str, config_path: Path) -> None:
    """Dispatch acquisition separately from offline evaluation, failing visibly."""
    spec = configuration(config_path)
    output = ROOT / spec["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if stage == "acquire":
        download_archives(ROOT / spec["labels_dir"], spec["archive_start"], spec["archive_end"])
        return
    if stage == "report":
        from v2.src.reporting import report

        report(output, spec)
        return
    ticker, labels, archives, rejected = inputs(spec)
    manifest = manifest_for(spec, config_path, archives)
    existing = output / "manifest.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if (
            previous["config_sha256"] != manifest["config_sha256"]
            or previous["ticker_sha256"] != manifest["ticker_sha256"]
            or previous["archives"] != archives
        ):
            raise ValueError(
                "Existing run has different protocol/sources; use a new output directory"
            )
    panel = validate_data(spec, output, ticker, labels, archives, rejected)
    save_json(output / "manifest.json", manifest)
    if stage == "validate-data":
        return
    if stage == "baselines":
        # Baselines do not require fitting RF/ridge or selecting them.
        validation = panel[
            panel.event_id.ge(spec["validation_start"]) & panel.event_id.lt(spec["test_start"])
        ]
        choices = {
            str(h): {
                "ewma_span": min(
                    spec["ewma_spans"],
                    key=lambda s: (
                        validation.loc[validation.horizon.eq(h), "target"]
                        - validation.loc[validation.horizon.eq(h), f"ewma_{s}"]
                    )
                    .abs()
                    .mean(),
                )
            }
            for h in spec["horizons"]
        }
        result, folds = forecast(panel, choices, spec, include_models=False)
        result.to_csv(
            output / "baseline_predictions.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        return
    choices = frozen_choices(
        panel, spec, output, manifest["config_sha256"], manifest["ticker_sha256"]
    )
    models(panel, ticker, labels, spec, output, choices)
    if sha256(ROOT / spec["ticker_path"]) != manifest["ticker_sha256"]:
        raise ValueError("Original input changed during execution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["acquire", "validate-data", "baselines", "models", "report"],
        required=True,
    )
    args = parser.parse_args()
    run(args.stage, args.config.resolve())


if __name__ == "__main__":
    main()
