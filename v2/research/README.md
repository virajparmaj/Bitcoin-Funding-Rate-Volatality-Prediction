# Original saved-prediction diagnostics

`reproduce_evidence.py` recomputes the original RF/persistence comparison and
SARIMAX diagnostic without training new models or changing original inputs:

```sh
v2/.venv/bin/python v2/research/reproduce_evidence.py
```

`evidence.json` and `saved_prediction_scores.csv` are small tracked scientific
reference artifacts. The demo exporter and validator use them to check saved-pair
arithmetic independently. They are not settlement forecasts; the completed
settlement study is documented in the [v2 README](../README.md).

The script also renders a local diagnostic PNG. That intermediate figure and the
long-form working assessment are ignored; neither is needed to run the demo or
the completed settlement study. Data and model redistribution permissions are
separate from the code's availability.
