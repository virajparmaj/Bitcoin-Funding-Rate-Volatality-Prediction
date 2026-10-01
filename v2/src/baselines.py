"""Forecasts that use only information available at the declared origin."""

from __future__ import annotations

import numpy as np


def baseline_predictions(train, test, ewma_span: int) -> dict:
    """Return aligned arrays; no target imputation or retrospective constants."""
    return {
        "indication": test.indication.to_numpy(),
        "last_settled": test.settled_1.to_numpy(),
        "ewma": test[f"ewma_{ewma_span}"].to_numpy(),
        "training_median": np.full(len(test), train.target.median()),
        "one_bp_reference": np.full(len(test), 0.0001),
    }
