"""Adversarial scientific checks for the event-aligned study (offline)."""

import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from v2.src.features import add_settled_history, ticker_features
from v2.src.forecast_models import estimator
from v2.src.funding_labels import load_labels
from v2.src.metrics import bootstrap_gain, loss_summary
from v2.src.settlement import build_origin_panel
from v2.src.splits import monthly_folds
from v2.src.timebase import normalize_ticker
from v2.src.validate import validate_panel


def utc(value):
    return pd.Timestamp(value, tz="UTC")


def source():
    event = utc("2024-01-02 08:00")
    arrival = pd.to_datetime(
        ["2024-01-02T02:59:00Z", "2024-01-02T04:00:00Z", "2024-01-02T04:00:00.000001Z"],
        format="mixed",
    )
    raw = pd.DataFrame(
        dict(
            local_timestamp=arrival.as_unit("us").asi8,
            funding_timestamp=[event.value // 1000] * 3,
            funding_rate=[0.0001, 0.0002, 0.0003],
            mark_price=[100.0, 101.0, 102.0],
            index_price=[100.0, 100.0, 100.0],
            open_interest=[1000.0, 1010.0, 1020.0],
        )
    )
    return normalize_ticker(raw)[0]


def labels():
    event = utc("2024-01-02 08:00")
    return pd.DataFrame(
        dict(
            event_id=[event],
            calc_time=[event],
            target=[0.00025],
            label_available_at=[event + pd.Timedelta(minutes=5)],
        )
    )


def make_archive(tmp_path, text):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample.csv", text)
    path.with_suffix(".zip.CHECKSUM").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  sample.zip"
    )
    return path


def test_exact_origin_allowed_future_microsecond_excluded():
    panel, excluded = build_origin_panel(source(), labels(), [4], 65)
    assert not len(excluded)
    assert panel.indication.iloc[0] == 0.0002
    assert panel.source_row_id.iloc[0] == 1
    validate_panel(panel, 65)


def test_stale_boundary_and_no_cross_event_borrow():
    ticker = source().iloc[:1].copy()
    panel, _ = build_origin_panel(ticker, labels(), [4], 61)
    assert len(panel) == 1
    panel, excluded = build_origin_panel(ticker, labels(), [4], 60)
    assert panel.empty and excluded.reason.iloc[0] == "stale_observation"
    ticker["event_id"] += pd.Timedelta(hours=8)
    panel, excluded = build_origin_panel(ticker, labels(), [4], 65)
    assert panel.empty and excluded.reason.iloc[0] == "missing_settlement_label"


def test_one_hour_delay_selects_earlier_actual_observation():
    panel, _ = build_origin_panel(source(), labels(), [4], 65, 60)
    assert panel.indication.iloc[0] == 0.0001
    assert panel.source_available_at.iloc[0] <= panel.origin.iloc[0]


def test_missing_label_never_imputed():
    panel, excluded = build_origin_panel(source(), labels().iloc[:0], [4], 65)
    assert panel.empty
    assert excluded.reason.iloc[0] == "missing_settlement_label"


def test_future_perturbation_preserves_earlier_features():
    original = source()
    altered = original.copy()
    altered.loc[altered.index[-1], ["funding_rate", "mark_price", "open_interest"]] = 999
    a, b = ticker_features(original), ticker_features(altered)
    assert_frame_equal(a.iloc[:2], b.iloc[:2])


def test_no_future_label_in_history():
    panel, _ = build_origin_panel(source(), labels(), [4], 65)
    with_history = add_settled_history(panel, labels(), [3])
    assert with_history.settled_1.isna().all()
    earlier = labels().copy()
    earlier["event_id"] -= pd.Timedelta(hours=8)
    earlier["label_available_at"] = panel.origin.iloc[0] + pd.Timedelta(microseconds=1)
    assert add_settled_history(panel, earlier, [3]).settled_1.isna().all()


def test_fold_drops_pre_fit_origin_and_unmatured_training_label():
    events = pd.to_datetime(["2023-01-31T16:00Z", "2023-02-01T00:00Z", "2023-02-01T08:00Z"])
    panel = pd.DataFrame(
        dict(
            event_id=events,
            origin=events - pd.Timedelta(hours=4),
            label_available_at=events + pd.Timedelta(minutes=5),
        )
    )
    cutoff, train, test = next(monthly_folds(panel, "2023-02-01T00:00Z", "2023-02-02T00:00Z"))
    assert len(train) == 1 and len(test) == 1
    assert (train.label_available_at < cutoff).all() and (test.origin >= cutoff).all()


def test_units_and_identical_baseline():
    frame = pd.DataFrame(dict(target=[0.0001], indication=[0.0], prediction=[0.0]))
    result = loss_summary(frame)
    assert result["mae_bp"] == 1 and result["gain_bp"] == 0 and result["mse_skill"] == 0
    frame["prediction"] = 0.0001
    assert loss_summary(frame)["gain_bp"] == 1
    frame["indication"] = 0.0001
    assert loss_summary(frame)["mse_skill"] is None


def test_calendar_bootstrap_deterministic_constant_and_gaps():
    events = pd.Series(pd.date_range("2023-01-01", periods=30, freq="2D", tz="UTC"))
    a = bootstrap_gain(events, np.ones(30), 200, 7, 42)
    b = bootstrap_gain(events, np.ones(30), 200, 7, 42)
    assert a == b and a["gain_low_bp"] == 1 and a["gain_high_bp"] == 1
    assert not bootstrap_gain(events[:1], [1], 200, 7, 42)["sufficient_blocks"]


def test_training_only_preprocessing_and_current_indication_legal():
    train = pd.DataFrame(dict(indication=[1.0, 2.0, np.nan], constant=[1.0, 1.0, 1.0]))
    fitted = estimator("ridge", {"alpha": 1.0}, {})
    fitted.fit(train, [1.0, 2.0, 3.0])
    before = fitted[0].statistics_.copy()
    fitted.predict(pd.DataFrame(dict(indication=[1e9], constant=[1.0])))
    np.testing.assert_array_equal(before, fitted[0].statistics_)


def test_archive_jitter_retained_and_checksum_checked(tmp_path):
    event = utc("2024-01-01 08:00")
    text = (
        f"calc_time,funding_interval_hours,last_funding_rate\n{event.value//1000000+25},8,0.0001\n"
    )
    path = make_archive(tmp_path, text)
    actual, manifest = load_labels([path], 5, 1)
    assert actual.event_id.iloc[0] == event
    assert actual.offset_seconds.iloc[0] == 0.025 and actual.target.iloc[0] == 0.0001
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        load_labels([path], 5, 1)


@pytest.mark.parametrize("offset,duplicate", [(2000, False), (0, True)])
def test_ambiguous_labels_fail(tmp_path, offset, duplicate):
    value = utc("2024-01-01 08:00").value // 1000000 + offset
    row = f"{value},8,0.0001\n"
    path = make_archive(
        tmp_path,
        "calc_time,funding_interval_hours,last_funding_rate\n" + row * (2 if duplicate else 1),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        load_labels([path], 5, 1)


def test_validation_rejects_future_source_and_duplicate_key():
    panel, _ = build_origin_panel(source(), labels(), [4], 65)
    changed = panel.copy()
    changed["source_available_at"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="Future"):
        validate_panel(changed, 65)
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(pd.concat([panel, panel]), 65)


def test_proxy_not_in_feature_allowlist():
    from v2.src.features import feature_columns

    assert "terminal_proxy" not in feature_columns("full")
    assert "target" not in feature_columns("full")
    assert "indication" in feature_columns("full")


def test_model_forecast_integration_keeps_targets_out_of_features():
    from v2.src.evaluation import forecast

    events = pd.date_range("2021-12-01", "2022-01-04", freq="8h", tz="UTC")
    panel = pd.DataFrame(
        dict(
            event_id=events,
            origin=events - pd.Timedelta(hours=4),
            source_available_at=events - pd.Timedelta(hours=5),
            source_row_id=np.arange(len(events)),
            age_minutes=60.0,
            label_available_at=events + pd.Timedelta(minutes=5),
            calc_time=events,
            history_available_at=events - pd.Timedelta(hours=7),
            target=0.0002,
            indication=0.0001,
            residual=0.0001,
            terminal_proxy=0.00019,
            terminal_proxy_at=events - pd.Timedelta(seconds=1),
            horizon=4,
            event_hour=events.hour,
        )
    )
    for lag in [1, 2, 3, 9]:
        panel[f"settled_{lag}"] = 0.0002
    panel["ewma_3"] = 0.0002
    spec = dict(
        horizons=[4],
        test_start="2022-01-01T00:00Z",
        test_end="2022-01-04T00:00Z",
        minimum_training_events=20,
        rf_trees=5,
        rf_jobs=1,
        seed=42,
    )
    choices = {
        "4": dict(
            ewma_span=3,
            ridge={"alpha": 1.0},
            history_ridge={"alpha": 1.0},
            rf={"max_depth": 3, "min_samples_leaf": 5},
        )
    }
    predictions, folds = forecast(panel, choices, spec, "indication")
    assert len(predictions.model.unique()) == 8
    assert (folds.min_test_origin >= folds.fit_time).all()
    assert (folds.max_train_label_available_at < folds.fit_time).all()
    assert np.isfinite(predictions.prediction).all()
    assert np.allclose(predictions.loc[predictions.model.eq("ridge"), "prediction"], 0.0002)


def test_common_events_removes_missing_variant_without_changing_targets():
    from v2.src.reporting import common_events

    rows = [
        dict(event_id=e, horizon=h, variant=v, model="ridge", target=e)
        for e in [1, 2]
        for h in [1, 4]
        for v in ["main", "delay"]
    ]
    data = pd.DataFrame(rows).iloc[:-1]
    result = common_events(data, [1, 4])
    assert set(result.event_id) == {1} and set(result.target) == {1}


def test_null_bootstrap_does_not_systematically_report_improvement():
    # Finite smoke check, not a calibration proof: symmetric null paths should not
    # all produce positive lower bounds just because they contain many events.
    rng = np.random.default_rng(17)
    events = pd.Series(pd.date_range("2023-01-01", periods=180, freq="D", tz="UTC"))
    positive = sum(
        bootstrap_gain(events, rng.normal(size=180), 300, 7, seed)["gain_low_bp"] > 0
        for seed in range(10)
    )
    assert positive <= 3
