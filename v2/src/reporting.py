"""Generate paper-facing tables from the immutable per-event forecast ledger."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .metrics import bootstrap_gain, inference_table, loss_summary
from .provenance import verify_report_inputs


def common_events(predictions: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Require every variant, model and horizon on identical event IDs."""
    expected = predictions[["variant", "model", "horizon"]].drop_duplicates().shape[0]
    if predictions.duplicated(["event_id", "variant", "model", "horizon"]).any():
        raise ValueError("Duplicate prediction key")
    if set(predictions.horizon.unique()) != set(horizons):
        raise ValueError("Missing forecast horizon")
    counts = predictions.groupby("event_id").size()
    return predictions[predictions.event_id.isin(counts[counts == expected].index)].copy()


def paired_effects(predictions: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """E2/E3/E3b differences use matched events and frozen predictions."""
    rows = []
    main = predictions[predictions.variant.eq("main")]
    for horizon in spec["horizons"]:
        for model in ("ridge", "rf"):
            reference = main[main.horizon.eq(horizon) & main.model.eq(model)].set_index("event_id")
            ref_error = abs(reference.target - reference.prediction) * 1e4
            ref_gain = abs(reference.target - reference.indication) * 1e4 - ref_error
            pairs = []
            for variant in ("indication", "history", "delay_60m"):
                other = (
                    predictions[
                        predictions.variant.eq(variant)
                        & predictions.horizon.eq(horizon)
                        & predictions.model.eq(model)
                    ]
                    .set_index("event_id")
                    .reindex(reference.index)
                )
                pairs.append(
                    (
                        f"main_error_minus_{variant}_error",
                        ref_error - abs(other.target - other.prediction) * 1e4,
                    )
                )
            proxy_gain = (
                abs(reference.terminal_proxy - reference.indication)
                - abs(reference.terminal_proxy - reference.prediction)
            ) * 1e4
            pairs.append(("proxy_gain_minus_settlement_gain", proxy_gain - ref_gain))
            if horizon != spec["primary_horizon"]:
                primary = (
                    main[main.horizon.eq(spec["primary_horizon"]) & main.model.eq(model)]
                    .set_index("event_id")
                    .reindex(reference.index)
                )
                primary_gain = (
                    abs(primary.target - primary.indication)
                    - abs(primary.target - primary.prediction)
                ) * 1e4
                pairs.append(("horizon_gain_minus_primary_gain", ref_gain - primary_gain))
            for comparison, difference in pairs:
                if not np.isfinite(difference).all():
                    raise ValueError("Unmatched/nonfinite paired contrast")
                rows.append(
                    dict(
                        horizon=horizon,
                        model=model,
                        comparison=comparison,
                        estimate_bp=float(difference.mean()),
                        n=len(difference),
                        **bootstrap_gain(
                            pd.Series(difference.index),
                            difference.to_numpy(),
                            spec["bootstrap_draws"],
                            7,
                            spec["seed"],
                        ),
                    )
                )
    return pd.DataFrame(rows)


def descriptive_slices(predictions: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Origin-known states and time slices; no confirmatory subgroup claims."""
    data = predictions[
        predictions.variant.eq("main") & predictions.model.isin(["ridge", "rf"])
    ].copy()
    data["state"] = np.where(
        abs(data.indication - 0.0001) <= spec["flat_tolerance"], "flat_1bp", "other"
    )
    data["quarter"] = data.event_id.dt.tz_localize(None).dt.to_period("Q").astype(str)
    rows = []
    for column in ("state", "quarter"):
        for (horizon, model, value), frame in data.groupby(["horizon", "model", column]):
            gains = (
                abs(frame.target - frame.indication) - abs(frame.target - frame.prediction)
            ).to_numpy() * 1e4
            rows.append(
                dict(
                    horizon=horizon,
                    model=model,
                    slice_type=column,
                    slice=value,
                    interpretation="descriptive_unadjusted",
                    **loss_summary(frame),
                    **bootstrap_gain(
                        frame.event_id, gains, spec["bootstrap_draws"], 7, spec["seed"]
                    ),
                )
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a small table without an optional tabulate dependency."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame[columns].itertuples(index=False, name=None):
        values = [f"{x:.6f}" if isinstance(x, float) else str(x) for x in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def figures(table: pd.DataFrame, output: Path) -> None:
    """Render measured error and incremental skill, never hypothetical curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = table[table.variant.eq("main") & table.block_days.eq(7)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = {"indication": "#47769c", "ridge": "#c77625", "rf": "#23836d"}
    for model in ("indication", "ridge", "rf"):
        frame = selected[selected.model.eq(model)].sort_values("horizon")
        axes[0].plot(frame.horizon, frame.mae_bp, marker="o", label=model, color=colors[model])
        if model != "indication":
            axes[1].plot(frame.horizon, frame.gain_bp, marker="o", label=model, color=colors[model])
            axes[1].fill_between(
                frame.horizon,
                frame.gain_low_bp,
                frame.gain_high_bp,
                alpha=0.15,
                color=colors[model],
            )
    axes[0].set_ylabel("MAE (bp; lower is better)")
    axes[1].set_ylabel("MAE gain vs indication (bp)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Hours before scheduled settlement")
        ax.set_xticks([1, 2, 4, 6])
        ax.legend()
    fig.suptitle("Retrospective matched-event forecasts; 95% pointwise calendar-block intervals")
    fig.savefig(output / "lead_time.png", dpi=180)
    plt.close(fig)


def verify_ledger(predictions, spec):
    """Require the complete declared experiment cross-product."""
    expected = {
        (v, m, h)
        for v in ["main", "indication", "history", "delay_60m"]
        for m in [
            "indication",
            "last_settled",
            "ewma",
            "training_median",
            "one_bp_reference",
            "ridge",
            "rf",
            "history_ridge",
        ]
        for h in spec["horizons"]
    }
    actual = set(
        map(tuple, predictions[["variant", "model", "horizon"]].drop_duplicates().to_numpy())
    )
    if actual != expected:
        raise ValueError("Incomplete declared experiment ledger")


def write_inference(data, spec, output):
    """Generate paired effects and a single secondary correction family."""
    table = inference_table(data, spec)
    contrasts = paired_effects(data, spec)
    # One declared secondary family covers model/horizon and paired ablation tests.
    secondary = (
        table.block_days.eq(7)
        & table.model.isin(["ridge", "rf"])
        & ~(
            table.variant.eq("main")
            & table.horizon.eq(spec["primary_horizon"])
            & table.model.eq("ridge")
        )
    )
    indices = table.index[secondary & table.p_value.notna()]
    adjusted = multipletests(
        list(table.loc[indices, "p_value"]) + list(contrasts.p_value), method="holm"
    )[1]
    table.loc[indices, "p_holm_secondary"] = adjusted[: len(indices)]
    contrasts["p_holm_secondary"] = adjusted[len(indices) :]
    table.to_csv(output / "scores.csv", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    return table


def write_supplements(data, predictions, spec, output):
    """Generate descriptive, proxy-label and age-sensitivity tables."""
    descriptive_slices(data, spec).to_csv(output / "state_quarter_scores.csv", index=False)
    proxy = []
    for (horizon, model), frame in data[data.variant.eq("main")].groupby(["horizon", "model"]):
        proxy.append(dict(horizon=horizon, model=model, **loss_summary(frame, "terminal_proxy")))
    pd.DataFrame(proxy).to_csv(output / "proxy_label_scores.csv", index=False)
    sensitivity = []
    for tolerance in (30, 60, 65):
        restricted = common_events(
            predictions[predictions.age_minutes <= tolerance], spec["horizons"]
        )
        for (horizon, model), frame in restricted[restricted.variant.eq("main")].groupby(
            ["horizon", "model"]
        ):
            sensitivity.append(
                dict(max_age_minutes=tolerance, horizon=horizon, model=model, **loss_summary(frame))
            )
    pd.DataFrame(sensitivity).to_csv(output / "staleness_scores.csv", index=False)


def write_narrative(data, predictions, table, spec, output):
    """Render the measured result and explicit limits into a manuscript-facing report."""
    horizon_table = table[
        table.variant.eq("main") & table.model.isin(["ridge", "rf"]) & table.block_days.eq(7)
    ]
    primary = table[
        table.variant.eq("main")
        & table.horizon.eq(spec["primary_horizon"])
        & table.block_days.eq(7)
    ]
    ridge = primary[primary.model.eq("ridge")].iloc[0]
    conclusion = (
        "The primary interval supports a positive incremental gain."
        if ridge.gain_low_bp > 0
        else "The primary interval does not establish a positive incremental gain."
    )
    sensitivity_path = output / "publication_delay_sensitivity.json"
    publication_note = "Publication-delay sensitivity has not been run."
    if sensitivity_path.exists():
        checks = json.loads(sensitivity_path.read_text())
        same = all(
            row["same_origin_keys"]
            and row["identical_settled_features"]
            and row["identical_fold_membership"]
            for row in checks
        )
        publication_note = (
            "0/5/15-minute publication assumptions produced identical origin keys, settled-history features and fold membership; no extra model refit was necessary."
            if same
            else "Publication-delay checks changed features or membership; model sensitivity remains required."
        )
    coverage = json.loads((output / "coverage.json").read_text())
    report_text = f"""# Executed settlement study

Protocol: {spec["protocol_version"]}. Retrospective evaluation; original data were previously inspected.

**{data.event_id.nunique()} common evaluation events** (from {predictions.event_id.nunique()} forecast events before the all-variant intersection), all four horizons, all models and availability variants. Primary comparison: residual ridge versus current indication at four hours.

{markdown_table(primary, ["model", "n", "mae_bp", "rmse_bp", "gain_bp", "gain_low_bp", "gain_high_bp"])}

Primary MAE reduction: **{ridge.gain_bp:.6f} bp**, 95% seven-day moving-calendar-block interval **[{ridge.gain_low_bp:.6f}, {ridge.gain_high_bp:.6f}] bp**. {conclusion}

## Secondary horizon results

{markdown_table(horizon_table, ["horizon", "model", "gain_bp", "gain_low_bp", "gain_high_bp", "p_holm_secondary"])}

The one-hour RF result is a declared secondary comparison, not a replacement for the primary four-hour ridge test. Pointwise intervals and multiplicity-adjusted tests answer different questions. Statistical evidence in this retrospective sample does not establish a deployable or economically material advantage.

## Data validation

{coverage["archives"]} verified archives; {coverage["archive_labels"]} official label records; {coverage["admitted_events"]} admitted events / {coverage["origin_rows"]} origins before evaluation splits. {coverage["rejected_ticker_rows"]} unusable ticker rows and {coverage["origin_exclusions"]} warm-up/eligibility origins were excluded explicitly. Maximum archive-to-schedule offset: {coverage["label_offset_max_seconds"]} seconds.

{publication_note}

## What was executed

- E0: full checksum-verified label history, event matching, availability checks and exclusions.
- E1/E2: monthly expanding forecasts, validation-only model selection and matched lead-time comparison.
- E3: fixed-hyperparameter indication/history/full feature fits and a one-hour source-delay variant.
- E3b: identical predictions rescored against terminal indications, plus paired differences in measured gain.
- E4: descriptive origin-known flat-state and calendar-quarter slices. These are not confirmatory subgroup discoveries.
- 3/7/14-day bootstrap sensitivity and 30/60/65-minute maximum-age sample sensitivity.

## Interpretation and limits

Positive gain means smaller MAE than the current indication. Errors are in basis points; regression error is not trading profit. Confidence intervals are conditional on the fitted forecasting procedures and do not erase retrospective protocol selection. Secondary model/horizon and paired contrasts share Holm-adjusted tests. Individual intervals are pointwise; no simultaneous practical-equivalence or family-wide unpredictability claim is made.

The one-hour delay shifts source availability, not just timestamp labels. Its staleness measures time since delayed availability; source availability, origin and variant are recorded, allowing receipt time to be reconstructed from the configured delay. Feature ablations reuse hyperparameters selected for the full feature model, so they measure fixed-configuration information removal rather than independently optimal reduced models.

Original hourly aggregation may combine fields from different messages. Its recorded arrival is an assumed conservative availability bound, not verified tick-level provenance. Label publication uses an assumed five-minute delay. Fresh-time replication, DAR replication, full seed/window/tail robustness and exact closest-paper novelty verification remain outstanding. This run does not establish publication priority, market efficiency or profitability.

The December 2019 warm-up archive returned HTTP 404; the implemented study starts in January 2020 and explicitly removes incomplete settled-history warm-up. Boundary events whose earliest origin precedes the monthly fit are excluded. The original feature file and v1 experiments are unchanged.

## Evidence

See `coverage.json`, `manifest.json`, `selection.json`, `validation_candidates.json`, `folds.csv`, `predictions.csv.gz`, `scores.csv`, `paired_contrasts.csv`, `proxy_label_scores.csv`, `state_quarter_scores.csv`, `staleness_scores.csv` and `lead_time.png`. Raw provider rows and the derived feature panel are not intended for redistribution; consult the release notes.
"""
    (output / "REPORT.md").write_text(report_text)


def report(output: Path, spec: dict) -> None:
    """Write E1–E4 results from verified saved forecasts."""
    verify_report_inputs(output, spec)
    predictions = pd.read_csv(
        output / "predictions.csv.gz",
        parse_dates=["event_id", "origin"],
        float_precision="round_trip",
    )
    verify_ledger(predictions, spec)
    data = common_events(predictions, spec["horizons"])
    table = write_inference(data, spec, output)
    write_supplements(data, predictions, spec, output)
    figures(table, output)
    write_narrative(data, predictions, table, spec, output)
