"""Read-only measurements supporting the research assessment (not a forecasting study).

Run from any directory with the existing v2 environment. Inputs are never modified;
derived JSON, tables and a diagnostic figure are written beside this script.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.src import audit, legacy

OUT = Path(__file__).resolve().parent


def main():
    paths = [ROOT / "data/normalized_datasets/binance_btc_perp.csv",
             ROOT / "results/predictions_RFR.csv",
             ROOT / "results/predictions_SARIMAX.csv"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    raw, rf, sar = [pd.read_csv(p) for p in paths]
    arrival = pd.to_datetime(raw.local_timestamp, unit="us", utc=True)
    funding_time = pd.to_datetime(raw.funding_timestamp, unit="us", utc=True)
    synthetic = pd.to_datetime(raw.timestamp, unit="us", utc=True)
    grouped = raw.groupby("funding_timestamp").funding_rate.agg(["size", "nunique", "first", "last"])
    drift = (synthetic - arrival).dt.total_seconds() / 3600
    gaps = arrival.diff().dt.total_seconds()
    preprocessed = legacy.preprocess_v1(raw)
    zeroed = legacy.zeroed_outlier_mask(preprocessed)
    backfilled = audit.backfill_audit(raw)
    recon = 3 * rf.funding_rate_ma3 - rf.funding_rate_lag1 - rf.funding_rate_lag2
    scores = []
    for name, prediction in [("RF saved predictions", rf.Predicted),
                             ("Current rate reconstructed (persistence)", recon),
                             ("Lag1 column (one observation older)", rf.funding_rate_lag1),
                             ("EMA3", rf.funding_rate_ema3)]:
        scores.append({"model": name, "n": len(rf), "r2": r2_score(rf.Actual, prediction),
                       "mae_bp": float(np.mean(abs(rf.Actual - prediction)) * 1e4),
                       "rmse_bp": float(np.sqrt(mean_squared_error(rf.Actual, prediction)) * 1e4),
                       "mse_native": mean_squared_error(rf.Actual, prediction)})
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "saved_prediction_scores.csv", index=False)

    # Verify target timing against the raw file, using the synthetic timestamp only
    # as a row-identity key. It is NOT used as a valid market-time variable.
    check = pd.DataFrame({"key": synthetic, "raw_current": raw.funding_rate,
                          "raw_next": raw.funding_rate.shift(-1), "arrival": arrival})
    join = pd.DataFrame({"key": pd.to_datetime(rf.timestamp, utc=True),
                         "recon": recon, "Actual": rf.Actual}).merge(
                             check, on="key", how="left", validate="one_to_one")
    valid = join.dropna(subset=["raw_current", "raw_next"])
    assert len(join) == len(rf)
    assert np.allclose(valid.recon, valid.raw_current, rtol=1e-6, atol=1e-12)
    assert np.allclose(valid.Actual, valid.raw_next, rtol=1e-6, atol=1e-12)
    base_mse = mean_squared_error(rf.Actual, recon)
    model_mse = mean_squared_error(rf.Actual, rf.Predicted)
    base_mae = float(np.mean(abs(rf.Actual - recon)))
    model_mae = float(np.mean(abs(rf.Actual - rf.Predicted)))

    result = {
        "purpose": "Artifact verification; not clean prospective validation or a settlement backtest",
        "input_sha256": hashes,
        "raw_shape": list(raw.shape),
        "arrival_start": str(arrival.min()), "arrival_end": str(arrival.max()),
        "synthetic_end": str(synthetic.max()),
        "funding_event_ids": len(grouped),
        "event_ids_with_multiple_distinct_rates": int((grouped["nunique"] > 1).sum()),
        "event_ids_with_one_distinct_rate": int((grouped["nunique"] == 1).sum()),
        "varying_event_share": float((grouped["nunique"] > 1).mean()),
        "first_to_last_event_rate_abs_change_mean_bp": float(abs(grouped["last"] - grouped["first"]).mean() * 1e4),
        "one_bp_rate_share_all_rows": float((raw.funding_rate == .0001).mean()),
        "null_counts": raw.isna().sum().astype(int).to_dict(),
        "drift_first_hours": float(drift.iloc[0]), "drift_last_hours": float(drift.iloc[-1]),
        "arrival_negative_differences": int((gaps < 0).sum()),
        "arrival_adjacent_gaps_over_90min": int((gaps > 5400).sum()),
        "arrival_gap_seconds_quantiles": gaps.quantile([0, .001, .01, .5, .99, 1]).to_dict(),
        "note_on_gaps": "Adjacent differences omit pairs containing the one missing arrival timestamp; flooring near-hour-end arrivals also creates bin collisions and must not be interpreted as exact duplicate messages.",
        "backfilled_cells_by_column": backfilled.backward_filled.astype(int).to_dict(),
        "rows_zeroed_by_legacy_outlier_path": int(zeroed.sum()),
        "zeroed_mean_abs_true_bp": float(preprocessed.loc[zeroed, "funding_rate"].abs().mean() * 1e4),
        "rf_alignment_total_rows": len(join), "rf_alignment_rows_with_both_raw_values": len(valid),
        "rf_current_max_abs_discrepancy": float(abs(valid.recon - valid.raw_current).max()),
        "rf_next_max_abs_discrepancy": float(abs(valid.Actual - valid.raw_next).max()),
        "rf_real_test_start": str(join.arrival.min()), "rf_real_test_end": str(join.arrival.max()),
        "rf_excess_mse_vs_current_rate_pct": float((model_mse / base_mse - 1) * 100),
        "rf_excess_mae_vs_current_rate_pct": float((model_mae / base_mae - 1) * 100),
        "rf_skill_vs_current_rate": float(1 - model_mse / base_mse),
        "rf_constant_test_columns": {c: int(rf[c].nunique()) for c in
            ["model1_direction_pred", "model2_volatility_h1", "predicted_funding_rate", "local_timestamp"]},
        "sarimax_n": len(sar), "sarimax_r2": float(r2_score(sar.Actual, sar.Predicted)),
        "sarimax_mae_bp_assuming_1e6_scaling": float(abs(sar.Actual - sar.Predicted).mean() / 1e6 * 1e4),
        "sarimax_units_note": "1e6 scaling is supported by notebook cell 17; saved file does not contain a units manifest.",
    }
    (OUT / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    data = scores.iloc[[1, 0, 2]]
    labels = ["Current-rate persistence", "Saved Random Forest", "Older lag1 baseline"]
    bars = ax.barh(labels, data.mae_bp, color=["#147d64", "#bd573e", "#8b99a6"])
    ax.invert_yaxis()
    for bar, value in zip(bars, data.mae_bp):
        ax.text(value + .004, bar.get_y() + bar.get_height()/2, f"{value:.4f}", va="center")
    ax.set_xlim(0, .205)
    ax.set_xlabel("Mean absolute error (basis points; lower is better)")
    ax.set_title("The baseline changes the interpretation of Model 3", loc="left", pad=18)
    fig.suptitle("7,797 saved next-row predictions • diagnostic only, not settlement forecasts", fontsize=10, y=1.05)
    fig.savefig(OUT / "baseline_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    print(scores.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
