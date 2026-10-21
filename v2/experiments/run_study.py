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

from v2.src.evaluation import forecast, select_ewma, tune
from v2.src.features import add_settled_history, ticker_features
from v2.src.funding_labels import download_archives, load_labels, sha256
from v2.src.settlement import build_origin_panel
from v2.src.timebase import normalize_ticker
from v2.src.validate import validate_panel
from v2.src.study_store import check_store, execute_stage, execution_identity

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
        git_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        python=platform.python_version(),
        packages={
            p: version(p)
            for p in ["numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "matplotlib"]
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


def frozen_choices(panel, spec, output, identity):
    """Record validation choices bound to the entire execution identity."""
    choices, ledger = tune(panel, spec)
    baseline = json.loads((output / "baseline_selection.json").read_text())
    if any(choices[h]["ewma_span"] != value["ewma_span"] for h, value in baseline.items()):
        raise ValueError("Baseline and model EWMA selection disagree")
    save_json(output / "validation_candidates.json", ledger)
    save_json(output / "selection.json", dict(identity=identity, choices=choices))
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


def load_panel(output):
    """Read exact serialized inputs shared by every downstream stage."""
    times = [
        "event_id",
        "origin",
        "source_available_at",
        "label_available_at",
        "history_available_at",
        "calc_time",
        "terminal_proxy_at",
    ]
    return pd.read_csv(
        output / "panel.csv.gz",
        parse_dates=times,
        date_format="mixed",
        float_precision="round_trip",
    )


def stage_action(stage, output, spec, manifest, ticker, labels, archives, rejected):
    """Produce a stage inside an unpublished transaction directory."""
    identity = execution_identity(manifest)
    if stage == "validate-data":
        validate_data(spec, output, ticker, labels, archives, rejected)
        save_json(output / "manifest.json", manifest)
        return
    panel = load_panel(output)
    if stage == "baselines":
        choices, ledger = select_ewma(panel, spec)
        result, folds = forecast(panel, choices, spec, include_models=False)
        result.to_csv(
            output / "baseline_predictions.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        folds.to_csv(output / "baseline_folds.csv", index=False)
        save_json(output / "baseline_selection.json", choices)
        save_json(output / "baseline_validation_candidates.json", ledger)
    elif stage == "models":
        choices = frozen_choices(panel, spec, output, identity)
        models(panel, ticker, labels, spec, output, choices)
    elif stage == "sensitivities":
        from v2.experiments.check_sensitivities import check

        check(output, spec, ticker, labels, identity)
    elif stage == "replay":
        from v2.experiments.verify_saved_forecasts import replay

        replay(output, spec, identity)
    elif stage == "report":
        from v2.src.reporting import report

        report(output, spec)


def run(stage: str, config_path: Path) -> None:
    """Reject changed evidence before writing; commit successful stages atomically."""
    spec = configuration(config_path)
    output = ROOT / spec["output_dir"]
    if stage == "acquire" and not (output / "manifest.json").exists():
        download_archives(ROOT / spec["labels_dir"], spec["archive_start"], spec["archive_end"])
        return
    ticker, labels, archives, rejected = inputs(spec)
    manifest = manifest_for(spec, config_path, archives)
    identity = execution_identity(manifest)
    check_store(output, identity)
    if stage == "acquire":
        download_archives(ROOT / spec["labels_dir"], spec["archive_start"], spec["archive_end"])
        return

    def recheck():
        _, _, current_archives, _ = inputs(spec)
        return execution_identity(
            manifest_for(configuration(config_path), config_path, current_archives)
        )

    execute_stage(
        output,
        stage,
        identity,
        lambda work: stage_action(stage, work, spec, manifest, ticker, labels, archives, rejected),
        recheck,
    )


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
