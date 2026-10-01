"""Scientific and transaction failures that must invalidate execution evidence."""

import json

import numpy as np
import pandas as pd
import pytest

from v2.experiments.check_sensitivities import compare_panels
from v2.src.evaluation import horizon_folds, select_ewma, tune
from v2.src.funding_labels import sha256
from v2.src.study_store import check_store, execute_stage


def snapshot(path):
    return {str(p.relative_to(path)): sha256(p) for p in path.rglob("*") if p.is_file()}


def commit(output, identity, stage="validate-data", action=None, recheck=None):
    execute_stage(
        output,
        stage,
        identity,
        action or (lambda work: (work / "panel.csv").write_text("target\n1\n")),
        recheck or (lambda: identity),
    )


@pytest.mark.parametrize("changed", ["config", "code", "archive", "ticker", "packages"])
def test_changed_identity_rejected_without_writes(tmp_path, changed):
    output = tmp_path / "run"
    original = {name: "original" for name in ["config", "code", "archive", "ticker", "packages"]}
    commit(output, original)
    before = snapshot(output)
    changed_identity = dict(original, **{changed: "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        commit(output, changed_identity)
    assert snapshot(output) == before


@pytest.mark.parametrize(
    "artifact", ["panel.csv", "selection.json", "publication_delay_sensitivity.json"]
)
def test_corrupt_evidence_aborts_before_other_writes(tmp_path, artifact):
    output = tmp_path / "run"
    commit(output, {}, action=lambda work: (work / artifact).write_text("original"))
    (output / artifact).write_text("changed")
    before = snapshot(output)
    with pytest.raises(ValueError, match="Corrupted"):
        commit(output, {})
    assert snapshot(output) == before


def test_failure_is_atomic_and_retry_and_completed_stage_are_safe(tmp_path):
    output = tmp_path / "run"

    def interrupted(work):
        (work / "partial.csv").write_text("unfinished")
        raise RuntimeError("fit failed")

    with pytest.raises(RuntimeError):
        commit(output, {}, action=interrupted)
    assert not output.exists()
    commit(output, {})
    before = snapshot(output)
    commit(output, {}, action=interrupted)
    assert snapshot(output) == before
    # A kill after canonical commit but before publishing links is recoverable.
    (output / "panel.csv").unlink()
    commit(output, {}, action=interrupted)
    assert snapshot(output) == before


def test_midrun_identity_change_never_commits(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="during execution"):
        commit(output, {}, recheck=lambda: {"code": "changed"})
    assert not output.exists()


def test_stage_cannot_modify_dependency_or_skip_prerequisite(tmp_path):
    output = tmp_path / "run"
    commit(output, {})
    before = snapshot(output)
    with pytest.raises(ValueError, match="before models"):
        commit(output, {}, stage="models")
    with pytest.raises(ValueError, match="overwrite"):
        commit(
            output,
            {},
            stage="baselines",
            action=lambda work: (work / "panel.csv").write_text("wrong"),
        )
    assert snapshot(output) == before


def test_legacy_run_is_preserved(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="Legacy"):
        check_store(tmp_path, {})
    assert snapshot(tmp_path) == before


def validation_fixture():
    # The 04:00 event's one-hour origin is after the monthly cutoff, but its
    # six-hour origin is before it: the whole event must be excluded.
    rows = []
    for event in pd.to_datetime(["2021-12-31T08:00Z", "2022-01-01T04:00Z", "2022-01-01T08:00Z"]):
        for horizon in [1, 6]:
            boundary = event.hour == 4
            rows.append(
                dict(
                    event_id=event,
                    horizon=horizon,
                    origin=event - pd.Timedelta(hours=horizon),
                    label_available_at=event + pd.Timedelta(minutes=5),
                    target=0.0,
                    ewma_3=100.0 if boundary else 0.0,
                    ewma_9=0.0 if boundary else 1.0,
                )
            )
    spec = dict(
        horizons=[1, 6],
        minimum_training_events=1,
        validation_start="2022-01-01T00:00Z",
        test_start="2023-01-01T00:00Z",
        test_end="2023-02-01T00:00Z",
        ewma_spans=[3, 9],
        ridge_alphas=[1.0],
        rf_depths=[3],
        rf_min_leaves=[5],
    )
    return pd.DataFrame(rows), spec


def test_ewma_and_models_share_monthly_event_cohort(monkeypatch):
    panel, spec = validation_fixture()
    choices, ledger = select_ewma(panel, spec)
    assert all(value["ewma_span"] == 3 for value in choices.values())
    assert all(row["n"] == 1 for row in ledger)
    cohorts = []

    def fake_model(model, params, train, test, columns, spec):
        cohorts.append(set(test.event_id))
        return np.zeros(len(test))

    monkeypatch.setattr("v2.src.evaluation.predict_model", fake_model)
    selected, candidates = tune(panel, spec)
    assert all(row["n"] == 1 for row in candidates)
    assert all(selected[h]["ewma_span"] == choices[h]["ewma_span"] for h in choices)
    assert all(c == {pd.Timestamp("2022-01-01T08:00Z")} for c in cohorts)
    assert all(len(test) == 1 for _, _, test in horizon_folds(panel, spec, 1, True))


def test_exact_delay_check_detects_tiny_change_and_training_membership():
    panel, spec = validation_fixture()
    original = compare_panels(panel, panel.copy(), ["ewma_3"], spec)
    assert all(original.values())
    changed = panel.copy()
    changed.loc[0, "ewma_3"] += 1e-12
    assert not compare_panels(panel, changed, ["ewma_3"], spec)["identical_settled_features"]
    changed = panel.copy()
    changed.loc[0, "label_available_at"] = pd.Timestamp("2022-01-01T00:01Z")
    assert not compare_panels(panel, changed, ["ewma_3"], spec)["identical_fold_membership"]


@pytest.mark.parametrize(
    "module,stage", [("check_sensitivities", "sensitivities"), ("verify_saved_forecasts", "replay")]
)
def test_helpers_forward_alternate_config(monkeypatch, tmp_path, module, stage):
    import importlib

    helper = importlib.import_module("v2.experiments." + module)
    config = tmp_path / "alternate.json"
    calls = []
    monkeypatch.setattr("sys.argv", [module, "--config", str(config)])
    monkeypatch.setattr(helper, "run", lambda s, p: calls.append((s, p)))
    helper.main()
    assert calls == [(stage, config.resolve())]


def test_report_rejects_stale_sensitivity_without_writing(tmp_path):
    from v2.src.provenance import verify_report_inputs
    from v2.src.study_store import digest

    identity = dict(
        config_sha256="config",
        protocol={},
        archives=[],
        ticker_sha256="ticker",
        code_sha256={},
        python="version",
        packages={},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(identity))
    (tmp_path / "selection.json").write_text(json.dumps({"identity": identity}))
    (tmp_path / "publication_delay_sensitivity.json").write_text(
        json.dumps(
            {
                "identity_sha256": digest(dict(identity, ticker_sha256="stale")),
            }
        )
    )
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="Stale"):
        verify_report_inputs(tmp_path, {})
    assert snapshot(tmp_path) == before


def test_report_requires_exact_invariance(tmp_path):
    from v2.src.provenance import verify_report_inputs
    from v2.src.study_store import digest

    identity = dict(
        config_sha256="config",
        protocol={},
        archives=[],
        ticker_sha256="ticker",
        code_sha256={},
        python="version",
        packages={},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(identity))
    (tmp_path / "selection.json").write_text(json.dumps({"identity": identity}))
    checks = [
        dict(
            label_delay_minutes=d,
            same_origin_keys=True,
            identical_settled_features=d != 15,
            identical_fold_membership=True,
        )
        for d in [0, 5, 15]
    ]
    for filename in ["publication_delay_sensitivity.json", "forecast_replay.json"]:
        (tmp_path / filename).write_text(
            json.dumps(dict(identity_sha256=digest(identity), checks=checks, status="passed"))
        )
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="Publication assumptions"):
        verify_report_inputs(tmp_path, {})
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("change,passes", [("roundoff", True), ("score", False), ("choice", False)])
def test_full_reproduction_comparison_tolerance_and_exact_choices(
    monkeypatch, tmp_path, change, passes
):
    from v2.experiments import compare_study_runs as comparator

    reference, candidate, work = (tmp_path / name for name in ["reference", "candidate", "work"])
    for directory in [reference, candidate, work]:
        directory.mkdir()
        (directory / "manifest.json").write_text("{}")
    record = dict(horizon=4, model="rf", params={"max_depth": 3}, mae_bp=0.25, n=1083)
    for directory in [reference, candidate]:
        (directory / "selection.json").write_text(json.dumps({"choices": {"4": {"alpha": 1.0}}}))
        (directory / "baseline_selection.json").write_text("{}")
        (directory / "coverage.json").write_text("{}")
        for name in ["validation_candidates.json", "baseline_validation_candidates.json"]:
            (directory / name).write_text(json.dumps([record]))
    if change in ["roundoff", "score"]:
        changed = dict(record, mae_bp=record["mae_bp"] + (1e-13 if change == "roundoff" else 1e-3))
        (candidate / "validation_candidates.json").write_text(json.dumps([changed]))
    else:
        (candidate / "selection.json").write_text(json.dumps({"choices": {"4": {"alpha": 100.0}}}))
    identity = {k: {} for k in ["archives", "ticker_sha256", "code_sha256", "python", "packages"]}
    configs = [
        comparator.ROOT / "v2/configs/test_reference.json",
        comparator.ROOT / "v2/configs/test_candidate.json",
    ]

    def inspect(config):
        directory = reference if config == configs[0] else candidate
        return dict(output_dir=str(directory)), identity, directory, {}

    monkeypatch.setattr(comparator, "inspect_run", inspect)
    monkeypatch.setattr(
        comparator, "execute_stage", lambda out, stage, ident, action, recheck: action(work)
    )
    if passes:
        comparator.compare(*configs)
        assert json.loads((work / "reproduction_check.json").read_text())["status"] == "passed"
    else:
        with pytest.raises((ValueError, AssertionError)):
            comparator.compare(*configs)
        assert not (work / "reproduction_check.json").exists()
