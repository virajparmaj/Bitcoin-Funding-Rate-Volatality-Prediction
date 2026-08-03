"""Paths and constants for the v2 research build.

Every value that v2 code depends on lives here. No module in ``v2.src`` may
hardcode a path, a window length, a fee, or a seed.

Path policy
-----------
The v1 submission is the *evidence being audited* and is read-only. This module
exposes v1 paths for reading and provides :func:`assert_writable` so that any
accidental write into a v1 directory fails loudly rather than silently
overwriting the artefact under audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

# ===========================================
# Directories
# ===========================================

V2_DIR: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = V2_DIR.parent

# --- v1, READ-ONLY ---
V1_DATA_CSV: Path = REPO_ROOT / "data" / "normalized_datasets" / "binance_btc_perp.csv"
V1_RESULTS_DIR: Path = REPO_ROOT / "results"
V1_PREDICTIONS_RFR_CSV: Path = V1_RESULTS_DIR / "predictions_RFR.csv"
V1_PREDICTIONS_SARIMAX_CSV: Path = V1_RESULTS_DIR / "predictions_SARIMAX.csv"
V1_NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"

# --- v2, WRITABLE ---
V2_DATA_DIR: Path = V2_DIR / "data"
V2_RESULTS_DIR: Path = V2_DIR / "results"
V2_NOTEBOOKS_DIR: Path = V2_DIR / "notebooks"

#: Directories v2 code is forbidden to write into.
PROTECTED_DIRS: tuple[Path, ...] = (
    V1_RESULTS_DIR,
    V1_NOTEBOOKS_DIR,
    REPO_ROOT / "data",
    REPO_ROOT / "models",
    REPO_ROOT / "utilities",
    REPO_ROOT / "dgieser3",
)

# ===========================================
# Domain constants
# ===========================================

#: Basis points per unit. Funding rates are quoted as decimals; 1e-4 == 1 bp.
BP: float = 1e4

#: Binance USD-M perpetual taker fee, one side, in basis points.
FEE_BP: float = 4.5

#: Cost of entering and exiting a hedged funding position (2 legs x 2 sides).
ROUND_TRIP_BP: float = 2 * 2 * FEE_BP  # 18.0 bp for a fully hedged pair

#: Cost of a single round trip on one instrument, used as the headline hurdle.
SINGLE_ROUND_TRIP_BP: float = 2 * FEE_BP  # 9.0 bp

#: Binance BTCUSDT perpetual settles funding every 8 hours (00/08/16 UTC).
SETTLEMENT_HOURS: int = 8
SETTLEMENT_UTC_HOURS: tuple[int, ...] = (0, 8, 16)

#: Native sampling frequency of the raw file.
RAW_SAMPLE_HOURS: int = 1

#: Global seed. Every stochastic estimator in v2 must take this explicitly.
SEED: int = 42

# ===========================================
# Evaluation constants
# ===========================================

#: Newey-West HAC lag count for Diebold-Mariano tests: one settlement day.
DM_HAC_LAGS: int = 8

#: Walk-forward embargo, in settlements. Must be >= 1 to purge the h=1 target.
EMBARGO_SETTLEMENTS: int = 1

#: Stress holdout: the March 2024 funding spike, carved out of normal scoring.
STRESS_WINDOW_START: datetime = datetime(2024, 3, 1, tzinfo=timezone.utc)
STRESS_WINDOW_END: datetime = datetime(2024, 4, 1, tzinfo=timezone.utc)

# ===========================================
# v1 replication constants (AUDIT ONLY)
# ===========================================
# These reproduce v1 behaviour inside legacy.py. They are never used by v2
# modelling code. Sources are cited as file:line against the v1 tree.

V1_SCALING_FACTOR: float = 1e6  # config.py DEFAULT_SCALING_FACTOR
V1_Z_SCORE_THRESHOLD: float = 3.0  # config.py DEFAULT_Z_SCORE_THRESHOLD
V1_TEST_SIZE: float = 0.2  # config.py DEFAULT_TEST_SIZE
V1_MA_WINDOW: int = 3  # utilities/functions.py:83
V1_VOL_WINDOW: int = 5  # utilities/functions.py:89
V1_ROLLING_STD_WINDOW: int = 24  # stat429_analysis_a.ipynb cell 22
V1_CLIP_LIMIT: float = 1e9  # utilities/functions.py:164
V1_N_ESTIMATORS: int = 100  # stat429_analysis_a.ipynb cell 30

#: dgieser3/research.ipynb cell 2 stamps a synthetic hourly clock from this epoch.
V1_SYNTHETIC_EPOCH: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc)

#: GARCH AIC reported in model_development.ipynb cell 12.
V1_GARCH_AIC: float = -732755.679


def assert_writable(path: Path) -> Path:
    """Raise if ``path`` falls inside a protected v1 directory.

    Args:
        path: Destination that v2 code intends to write.

    Returns:
        The same path, unchanged, when the write is permitted.

    Raises:
        PermissionError: If the destination is inside a protected v1 directory.
    """
    resolved = Path(path).resolve()
    for protected in PROTECTED_DIRS:
        if resolved == protected or protected in resolved.parents:
            raise PermissionError(
                f"Refusing to write to protected v1 path: {resolved}. "
                f"v2 outputs belong under {V2_RESULTS_DIR}."
            )
    return resolved


def ensure_dir(path: Path) -> Path:
    """Create a v2 output directory on demand, after checking it is permitted.

    ``V2_DATA_DIR`` is not tracked by git — the root ``.gitignore`` carries an
    unanchored ``data/`` pattern, which git applies at any depth, so the directory
    cannot be checked in via a placeholder file. It must therefore be created at
    runtime by whichever phase first writes derived data.

    Args:
        path: Directory to create.

    Returns:
        The resolved directory path.

    Raises:
        PermissionError: If the destination is inside a protected v1 directory.
    """
    resolved = assert_writable(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
