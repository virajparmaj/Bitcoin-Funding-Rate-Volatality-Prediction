"""Guard the hard constraint: v2 must never write into the v1 tree.

The rest of the test suite arrives in Phase 3 alongside ``features.py`` and
``validate.py``. This module exists from Phase 0 because it protects the one invariant
that, if broken, destroys the evidence the whole build is auditing.
"""

from __future__ import annotations

import pytest

from src import config


@pytest.mark.parametrize(
    "protected",
    [
        config.V1_RESULTS_DIR / "predictions_RFR.csv",
        config.V1_RESULTS_DIR / "anything.csv",
        config.V1_DATA_CSV,
        config.V1_NOTEBOOKS_DIR / "model_development.ipynb",
        config.REPO_ROOT / "utilities" / "functions.py",
        config.REPO_ROOT / "models" / "model3.py",
        config.REPO_ROOT / "dgieser3" / "research.ipynb",
    ],
)
def test_assert_writable_rejects_v1_paths(protected) -> None:
    """Every v1 path under audit must be refused."""
    with pytest.raises(PermissionError):
        config.assert_writable(protected)


@pytest.mark.parametrize(
    "allowed",
    [
        config.V2_RESULTS_DIR / "baselines.csv",
        config.V2_DATA_DIR / "settlement_panel.parquet",
        config.V2_DIR / "results" / "nested" / "out.json",
    ],
)
def test_assert_writable_allows_v2_paths(allowed) -> None:
    """v2's own output directories must be permitted."""
    assert config.assert_writable(allowed) == allowed.resolve()


def test_v1_inputs_exist_and_are_readable() -> None:
    """The audit depends on these files being present and unmodified."""
    assert config.V1_DATA_CSV.is_file()
    assert config.V1_PREDICTIONS_RFR_CSV.is_file()
    assert config.V1_PREDICTIONS_SARIMAX_CSV.is_file()


def test_ensure_dir_refuses_protected_and_creates_v2(tmp_path) -> None:
    """ensure_dir must apply the same guard as assert_writable before creating."""
    with pytest.raises(PermissionError):
        config.ensure_dir(config.V1_RESULTS_DIR / "new_subdir")
    assert not (config.V1_RESULTS_DIR / "new_subdir").exists()

    created = config.ensure_dir(tmp_path / "nested" / "out")
    assert created.is_dir()


def test_v2_data_dir_is_git_ignored() -> None:
    """The derived-data directory must never be committed.

    Guards a subtle failure: the root .gitignore's unanchored ``data/`` pattern already
    excludes v2/data at any depth, so a tracked placeholder there is impossible.
    """
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", str(config.V2_DATA_DIR / "anything.parquet")],
        cwd=config.REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, "v2/data must be git-ignored"


def test_round_trip_cost_matches_fee() -> None:
    """The economics hurdle must derive from FEE_BP, not be hardcoded separately."""
    assert config.SINGLE_ROUND_TRIP_BP == 2 * config.FEE_BP
    assert config.ROUND_TRIP_BP == 4 * config.FEE_BP
