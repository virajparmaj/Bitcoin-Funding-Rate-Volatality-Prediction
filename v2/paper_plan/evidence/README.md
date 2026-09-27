# Evidence index

All measurements here concern diagnostic verification or feasibility. They are not completed E1–E4 forecast experiments.

| File | Role |
|---|---|
| [evidence.json](evidence.json) | Frozen copy of prior repository measurements and input hashes |
| [saved_prediction_scores.csv](saved_prediction_scores.csv) | Corrected saved-artifact baseline comparison |
| [baseline_comparison.png](baseline_comparison.png) | Existing diagnostic chart |
| [archive_probe.json](archive_probe.json) | Three public source URLs, hashes, headers and retrieval outcomes |
| `BTCUSDT-fundingRate-*.zip` and `.CHECKSUM` | Three original public source samples, checksum verified |
| [check_feasibility.py](check_feasibility.py) | Offline reproducible origin coverage and sample label comparison |
| [feasibility.json](feasibility.json) | Computed feasibility numbers and caveats |
| [run_manifest.json](run_manifest.json) | Environment, exact source/script hashes and executed-check summary |
| [document_validation.json](document_validation.json) | Package link, syntax, snapshot and source-integrity checks |

The feasibility probe was deliberately limited to January 2020, June 2022 and March 2024. It must not be presented as random sampling, full label validation or a held-out test. Source downloads contained 276 labels and 275 had matching pre-event indications. Archive normalization preserves actual calc times and uses an explicit bounded candidate event mapping.

To refresh source samples, fetch each URL and its `.CHECKSUM` URL from `archive_probe.json`, compare SHA-256 before opening the ZIP, then rerun the offline script. Changed source checksums require a new manifest/version; do not overwrite historical evidence silently. The official archive can revise files.
