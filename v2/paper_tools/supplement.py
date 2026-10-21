"""Run fixed post-result sensitivities without modifying the original study.

Usage: python -m v2.paper_tools.supplement --config v2/configs/paper_supplement_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from v2.experiments.run_study import load_panel
from v2.src.evaluation import horizon_folds, predict_model
from v2.src.features import feature_columns
from v2.src.funding_labels import sha256
from v2.src.metrics import bootstrap_gain, loss_summary
from v2.src.reporting import common_events, verify_ledger
from v2.src.study_store import check_store, digest, execution_identity

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["event_id", "horizon", "model"]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def verify_parent(parent: Path) -> dict:
    """Check all receipts, numerical dependencies, raw sources and frozen code."""
    manifest = json.loads((parent / "manifest.json").read_text())
    check_store(parent, execution_identity(manifest))
    for filename, expected in manifest["code_sha256"].items():
        if sha256(ROOT / filename) != expected:
            raise ValueError(f"Frozen code changed: {filename}")
    if sha256(ROOT / manifest["protocol"]["ticker_path"]) != manifest["ticker_sha256"]:
        raise ValueError("Ticker source changed")
    for archive in manifest["archives"]:
        path = ROOT / manifest["protocol"]["labels_dir"] / archive["file"]
        if sha256(path) != archive["sha256"] or sha256(path.with_suffix(".zip.CHECKSUM")) != archive["checksum_sha256"]:
            raise ValueError("Raw archive or checksum changed")
    current = {p: version(p) for p in manifest["packages"]}
    if current != manifest["packages"] or platform.python_version() != manifest["python"]:
        raise ValueError("Numerical environment changed; use a separately reproduced parent")
    return manifest


def restrict_training(train, cutoff, months=None):
    """Calendar-window restriction retains label maturity and never uses test data."""
    if not (train.label_available_at < cutoff).all():
        raise ValueError("Unmatured training label")
    if months is None:
        return train
    if months <= 0:
        raise ValueError("Training window must be positive")
    return train[train.event_id.ge(cutoff - pd.DateOffset(months=months))]


def calibration_prediction(train, test, name):
    if name == "residual_mean":
        correction = train.residual.mean()
    elif name == "residual_median":
        correction = train.residual.median()
    else:
        raise ValueError("Unknown calibration baseline")
    return test.indication.to_numpy() + correction


def fit_sensitivities(panel, choices, spec, config):
    """Save each forecast and fit membership before any score is calculated."""
    outputs, folds = [], []
    variants = [(n, None, spec["seed"], [n]) for n in config["calibration_baselines"]]
    variants += [(f"window_{m}m", m, spec["seed"], ["ridge", "rf"]) for m in config["window_months"]]
    variants += [(f"seed_{s}", None, s, ["rf"]) for s in config["additional_rf_seeds"]]
    for variant, months, seed, models in variants:
        for horizon in spec["horizons"]:
            for cutoff, training, test in horizon_folds(panel, spec, horizon):
                train = restrict_training(training, cutoff, months)
                if len(train) < spec["minimum_training_events"]:
                    raise ValueError("Insufficient sensitivity training history")
                if not (test.origin >= cutoff).all():
                    raise ValueError("Test origin precedes fit")
                for model in models:
                    prediction = (calibration_prediction(train, test, model)
                                  if model in config["calibration_baselines"] else
                                  predict_model(model, choices[str(horizon)][model], train, test,
                                                feature_columns("full"), dict(spec, seed=seed)))
                    if not np.isfinite(prediction).all():
                        raise ValueError("Nonfinite sensitivity forecast")
                    result = test[["event_id", "origin", "horizon", "target", "indication",
                                   "label_available_at", "source_row_id"]].copy()
                    result["prediction"], result["model"] = prediction, model
                    result["variant"], result["fit_time"] = variant, cutoff
                    outputs.append(result)
                    folds.append(dict(variant=variant, model=model, horizon=horizon, fit_time=cutoff,
                                      train_n=len(train), test_n=len(test), seed=seed,
                                      min_train_event=train.event_id.min(),
                                      max_train_label_available_at=train.label_available_at.max(),
                                      min_test_origin=test.origin.min(),
                                      train_keys_sha256=digest(train.event_id.astype(str).tolist()),
                                      test_keys_sha256=digest(test.event_id.astype(str).tolist())))
            print(f"Completed {variant}, horizon {horizon}", flush=True)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(folds)


def score_frame(frame, config):
    gains = (abs(frame.target - frame.indication) - abs(frame.target - frame.prediction)).to_numpy() * 1e4
    return dict(**loss_summary(frame), **bootstrap_gain(frame.event_id, gains,
                config["bootstrap_draws"], config["block_days"], config["bootstrap_seed"]))


def simultaneous_bounds(frame, config):
    """Unstudentized maximum-deviation block bounds for a fixed eight-test family.

    Joint resampling keeps horizon/model dependence and empty calendar days.
    These are approximate conditional bounds, not a prospective equivalence test.
    """
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate family member/event")
    data = frame.copy()
    data["gain"] = (abs(data.target-data.indication)-abs(data.target-data.prediction))*1e4
    wide = data.pivot(index="event_id", columns=["horizon", "model"], values="gain").sort_index()
    if wide.empty or wide.isna().any().any():
        raise ValueError("Simultaneous family requires matched events")
    days = wide.index.floor("D")
    calendar = pd.date_range(days.min(), days.max(), freq="D")
    sums = wide.groupby(days).sum().reindex(calendar, fill_value=0).to_numpy()
    count = pd.Series(1, index=days).groupby(level=0).sum().reindex(calendar, fill_value=0).to_numpy()
    a = np.column_stack([sums, count])
    cumulative = np.vstack([np.zeros(a.shape[1]), a.cumsum(axis=0)])
    block, draws = config["block_days"], config["bootstrap_draws"]
    if len(a) < 2 * block:
        raise ValueError("Insufficient calendar blocks")
    full, rem = divmod(len(a), block)
    rng = np.random.default_rng(config["bootstrap_seed"])
    starts = rng.integers(0, len(a)-block+1, size=(draws, full))
    totals = (cumulative[starts+block]-cumulative[starts]).sum(axis=1)
    if rem:
        starts = rng.integers(0, len(a)-rem+1, size=draws)
        totals += cumulative[starts+rem]-cumulative[starts]
    totals = totals[totals[:, -1] > 0]
    estimate = wide.mean().to_numpy()
    samples = totals[:, :-1]/totals[:, -1:]
    radius = np.quantile(np.max(abs(samples-estimate), axis=1), .95)
    return pd.DataFrame([dict(horizon=h, model=m, gain_bp=estimate[i],
                              simultaneous_low_bp=estimate[i]-radius,
                              simultaneous_high_bp=estimate[i]+radius, family_size=len(estimate),
                              n=len(wide), confidence=.95)
                         for i, (h, m) in enumerate(wide.columns)])


def write_analyses(predictions, extra, panel, spec, config, output):
    verify_ledger(predictions, spec)
    matched = common_events(predictions, spec["horizons"])
    common_ids = set(matched.event_id)
    main = predictions[predictions.variant.eq("main")]
    rows = []
    for cohort, data in [("all_main", main), ("all_variant_matched", matched[matched.variant.eq("main")])]:
        for (h, m), frame in data.groupby(["horizon", "model"]):
            rows.append(dict(cohort=cohort, horizon=h, model=m, **score_frame(frame, config)))
    pd.DataFrame(rows).to_csv(output / "cohort_scores.csv", index=False)
    rows = []
    for (v, h, m), frame in extra[extra.event_id.isin(common_ids)].groupby(["variant", "horizon", "model"]):
        rows.append(dict(variant=v, horizon=h, model=m, **score_frame(frame, config)))
    pd.DataFrame(rows).to_csv(output / "sensitivity_scores.csv", index=False)
    central = matched[matched.variant.eq("main") & matched.model.isin(["ridge", "rf"])]
    simultaneous_bounds(central, config).to_csv(output / "simultaneous_bounds.csv", index=False)
    # Frozen pre-evaluation threshold, never optimized on evaluation outcomes.
    threshold = panel[panel.event_id.lt(spec["test_start"])].drop_duplicates("event_id").target.abs().quantile(config["tail_training_quantile"])
    tails, quarters = [], []
    for (h, m), frame in central.groupby(["horizon", "model"]):
        for name, subset in [("all", frame), ("above_training_q99", frame[frame.target.abs()>threshold]),
                             ("at_or_below_training_q99", frame[frame.target.abs()<=threshold])]:
            if len(subset):
                tails.append(dict(horizon=h, model=m, subset=name, threshold_bp=threshold*1e4,
                                  interpretation="outcome_conditioned_descriptive", **score_frame(subset, config)))
        q = frame.event_id.dt.tz_localize(None).dt.to_period("Q").astype(str)
        for quarter in sorted(q.unique()):
            subset = frame[q.ne(quarter)]
            quarters.append(dict(horizon=h, model=m, omitted_quarter=quarter,
                                 **score_frame(subset, config)))
    pd.DataFrame(tails).to_csv(output / "tail_scores.csv", index=False)
    pd.DataFrame(quarters).to_csv(output / "leave_quarter_out.csv", index=False)
    write_json(output / "cohort_counts.json", dict(main_events=main.event_id.nunique(),
               matched_events=len(common_ids), removed_events=sorted(str(x) for x in set(main.event_id)-common_ids),
               tail_threshold_bp=threshold*1e4))


def run(config_path):
    config = json.loads(config_path.read_text())
    parent, output = ROOT / config["parent"], ROOT / config["output"]
    if not output.resolve().is_relative_to((ROOT / "v2/paper_package").resolve()):
        raise ValueError("Paper output must stay in ignored v2/paper_package")
    manifest = verify_parent(parent)
    identity = dict(config=config, parent_identity=digest(execution_identity(manifest)),
                    code={str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    if output.exists():
        receipt = json.loads((output / "receipt.json").read_text())
        if receipt['identity'] != identity or any(sha256(output/f)!=s for f,s in receipt['outputs'].items()):
            raise ValueError("Changed completed supplement; use a new versioned output")
        print("Verified completed supplement", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    panel = load_panel(parent)
    choices = json.loads((parent / "selection.json").read_text())["choices"]
    predictions = pd.read_csv(parent / "predictions.csv.gz", parse_dates=["event_id", "origin"], float_precision="round_trip")
    with tempfile.TemporaryDirectory(prefix=".supplement-", dir=output.parent) as tmp:
        work = Path(tmp)
        extra, folds = fit_sensitivities(panel, choices, manifest['protocol'], config)
        extra.to_csv(work / "predictions.csv.gz", index=False, compression={"method":"gzip","mtime":0})
        folds.to_csv(work / "folds.csv", index=False)
        write_analyses(predictions, extra, panel, manifest['protocol'], config, work)
        verify_parent(parent)
        if identity['code'] != {str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))} or json.loads(config_path.read_text()) != config:
            raise ValueError("Supplement code or config changed during run")
        write_json(work / "receipt.json", dict(identity=identity, outputs={p.name:sha256(p) for p in sorted(work.iterdir())}))
        os.rename(work, output)
    print(f"Completed supplement: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("v2/configs/paper_supplement_v1.json"))
    run(parser.parse_args().config)
