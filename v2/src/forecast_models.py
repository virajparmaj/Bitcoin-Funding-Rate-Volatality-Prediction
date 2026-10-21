"""Training-only preprocessing and compact residual/history estimators."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def estimator(model: str, params: dict, spec: dict):
    """Construct an unfitted estimator; never fill outcomes or fit transforms globally."""
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if model in ("ridge", "history_ridge"):
        return make_pipeline(imputer, StandardScaler(), Ridge(alpha=params["alpha"]))
    if model == "rf":
        return make_pipeline(
            imputer,
            RandomForestRegressor(
                n_estimators=spec["rf_trees"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=1.0,
                random_state=spec["seed"],
                n_jobs=spec["rf_jobs"],
            ),
        )
    raise ValueError(f"Unknown model {model}")
