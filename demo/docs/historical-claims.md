# Historical model claims

These excerpts preserve narrative claims shown in the demo. They are not verified
model scores or current setup instructions. They were captured in the original
demo export; notebook outputs and saved-prediction arithmetic take precedence.

The complete working notes are local-only. These selected scientific definitions
and conflicting claims remain here so contributors can reproduce the demo.
Unrelated interview analogies and incomplete trailing items are omitted.

## notes-classification

Original source: `notes/09_results_and_metrics.md`, lines 46–56.

> ## Model 1 Direction Prediction
> Confirmed from report:
> | Model | Accuracy | F1 | ROC AUC | Interpretation |
> |---|---:|---:|---:|---|
> | Logistic Regression | 0.8026 | 0.6589 | N/A | Fast, interpretable baseline |
> | Random Forest initial | 0.7599 | 0.5459 | 0.835 | Nonlinear but needed tuning |
> | Random Forest tuned | 0.7738 | 0.7142 | 0.8718 | Best directional model by F1/ROC AUC |
>
> Note:
> - Notebook summary also says tuned RF accuracy `78.75%`; report table says `0.7738`. Treat as minor metric discrepancy needing verification.

## notes-analysis

Original source: `notes/09_results_and_metrics.md`, lines 35–45.

> ## Analysis A Baselines
> Confirmed from report:
> | Model | R2 | MSE | Note |
> |---|---:|---:|---|
> | Linear Regression 1 | 0.132 | 5.36e-08 | Weak linear baseline |
> | Linear Regression 2 | 0.284 | 4.42e-08 | Better with volatility/std feature |
> | Random Forest | 0.936 | 9.34e-09 | Best Analysis A model in report |
>
> Confirmed from notebook PDF:
> - Random Forest R2 about `0.936258`.

## notes-garch

Original source: `notes/09_results_and_metrics.md`, lines 57–69.

> ## Model 2 Volatility Prediction
> Confirmed from report:
> - Best model: `GJR-GARCH(1,1) AR(2)`.
> - Log-likelihood about `366385`.
> - AIC about `-732756`.
> - BIC about `-732696`.
> - More exact text: AIC `-732755.679`, BIC `-732695.682`.
>
> Interpretation:
> - Captures volatility clustering.
> - AR(2) terms align with PACF evidence around lags 1 and 2.
> - Forecasted variance rose over the next five steps, indicating increasing short-term uncertainty.

## notes-regression

Original source: `notes/09_results_and_metrics.md`, lines 70–82.

> ## Model 3 Exact Funding Rate
> Confirmed from report table:
> | Model 3 approach | R2 | MSE | Note |
> |---|---:|---:|---|
> | Random Forest Regressor | 0.96 | 0.0000023 | Strong ML baseline |
> | SARIMAX with exogenous variables | 0.925 | 0.0000112 | Integrates direction and volatility |
>
> Needs verification:
> - Report narrative also mentions R2 up to about `97.5%`.
> - Current `predictions_RFR.csv` recomputes near R2 `0.975`.
> - Current `predictions_SARIMAX.csv` appears scale-mismatched and does not match report metrics.

## notes-features

Original source: `notes/05_feature_engineering.md`, lines 11–43.

> ## Feature Families Used
> Confirmed from code and notebooks:
> - Lag features:
>   - `funding_rate_lag1`
>   - `funding_rate_lag2`
>   - `open_interest_lag1`
>   - `mark_price_lag1`
> - Rolling features:
>   - `funding_rate_ma3`
>   - `funding_rate_ma5`
>   - rolling standard deviation / volatility proxy
> - EMA features:
>   - `funding_rate_ema3`
>   - `funding_rate_ema5`
> - Rate-of-change features:
>   - `funding_rate_roc1`
>   - `funding_rate_roc3`
>   - `open_interest_roc`
> - Interaction terms:
>   - `interaction1 = funding_rate_lag1 * funding_rate_lag2`
>   - `interaction2 = funding_rate_ma3 / funding_rate_lag1`
>   - `interaction3 = mark_price_lag1 * funding_rate_ma3`
> - Time features:
>   - `hour`, `day`, `month`
>   - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos`
>
> ## Why These Features Were Used
> - Lags capture short-term memory.
> - Moving averages and EMAs smooth noisy behavior.
> - ROC features capture momentum and direction of change.
> - Volatility features capture recent uncertainty.
> - Interactions capture nonlinear relationships that linear models may miss.
> - Cyclical time features avoid treating hour 23 and hour 0 as far apart.

## notes-models

Original source: `notes/06_ml_models.md`, lines 11–78.

> ## Model 1: Direction Prediction
> Goal: predict whether the next funding rate moves up or down.
>
> Models:
> - Logistic Regression:
>   - Baseline classifier.
>   - Interpretable and fast.
>   - Assumes a mostly linear relationship between features and log-odds.
>   - Report result: accuracy about `0.8026`, F1 about `0.6589`; ROC AUC not reported in table.
> - Random Forest Classifier:
>   - Nonlinear classifier.
>   - Handles interactions and threshold effects.
>   - Report result: tuned RF F1 about `0.7142`, ROC AUC about `0.8718`; report text also notes F1 about `71.27%`.
>   - Chosen because F1/recall behavior was better for the imbalanced direction task.
>
> ## Model 2: Volatility Prediction
> Goal: estimate uncertainty/risk intensity, not exact direction.
>
> Models:
> - GARCH(1,1): baseline volatility clustering.
> - EGARCH: asymmetric shock modeling.
> - GJR-GARCH: leverage/asymmetric volatility effects.
> - GJR-GARCH(1,1) AR(2): best report-backed model by AIC/BIC.
>
> Report-backed result:
> - Best model: `GJR-GARCH(1,1) AR(2)`.
> - AIC about `-732755.679`.
> - BIC about `-732695.682`.
> - Forecasted conditional variances increased over five steps, indicating rising uncertainty.
>
> ## Model 3: Exact Funding-Rate Prediction
> Goal: predict the next continuous funding-rate value.
>
> Models:
> - Random Forest Regressor:
>   - Strong nonlinear baseline.
>   - Report table: RFR R2 `0.96`, MSE `0.0000023`.
>   - Current `predictions_RFR.csv` recomputes near R2 `0.975`; needs verification against report scaling and split.
> - SARIMAX:
>   - Time-series regression with exogenous variables.
>   - Report table: SARIMAX R2 `0.925`, MSE `0.0000112`.
>   - Report narrative also mentions about `97.5%` R2 in some places.
>   - Current `predictions_SARIMAX.csv` appears scale-mismatched; needs verification.
>
> ## Baseline Analysis A
> - Linear Regression 1:
>   - Report/notebook result: R2 about `0.132`, MSE about `5.36e-08`.
> - Linear Regression 2:
>   - Report/notebook result: R2 about `0.284`, MSE about `4.42e-08`.
> - Random Forest:
