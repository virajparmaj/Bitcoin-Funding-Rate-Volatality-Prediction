# Local evidence export

From the repository root, using the demo's isolated Python environment (Python
3.10 or later; standard library only, no Python packages to install):

```sh
python3 -m venv demo/.venv
demo/.venv/bin/python demo/scripts/export_research_content.py
demo/.venv/bin/python demo/scripts/export_evidence.py
demo/.venv/bin/python demo/scripts/validate_evidence.py
```

The exporter reads the repository's raw CSV and saved prediction artifacts. It
does not import research modules, load serialized models, execute notebook cells,
run experiments, or alter research files. All output is confined to
`demo/public/data/`:

- `evidence.json`: dataset counts and missingness, seven selected raw preview
  rows, scored RF/baseline records, a separate SARIMAX diagnostic, provenance,
  source excerpts, caveats, and explicit derivations.
- `series.json`: all 7,797 RF saved pairs and retrospective baseline values in
  native decimal-rate units, plus 1,752 raw daily summaries.
- `manifest.json`: source SHA-256 digests, output hashes, generator hash, evidence
  status definitions, and public-release status.

The sibling `export_research_content.py` owns `research-content.json`, which
contains task definitions, feature selections, stored notebook results and source
excerpts. Run it first. The evidence exporter includes its output and source
hashes in the manifest, and refuses to certify stale source hashes.

Outputs are deterministic: they contain no generation clock or machine-specific
paths. The validator checks byte-identical regeneration, paired metrics against
the existing research score table, raw counts, target alignment, native-to-bp
conversion, missing-day behavior, preserved daily extremes, output hashes, write
boundaries, and unchanged tracked research-file hashes before/after execution.

RF chart rows keep the CSV order and a zero-based saved-row index. They do not
pretend the synthetic timestamp is verified UTC. The optional target alignment
check uses the synthetic timestamp **only as a unique row key** to compare saved
values to raw current and next observations. Metrics include every saved RF pair;
the alignment check has 7,793 complete raw pairs because four rows lack a raw
current or next rate. This is an artifact diagnostic, not a new validation split.

Daily data use valid `local_timestamp` values in UTC, excluding one raw row with
missing arrival time. Each variable retains mean, exact min/max, and valid and
missing counts. Empty days remain present with null values. The reference of 24
rows per day is **nominal**: a shortfall or excess does not establish real missing
observations because near-hour arrival boundaries may create bin collisions.
No raw values are filled, interpolated, or removed as outliers.

Funding rates are exported in native decimal units; rate × 10,000 gives basis
points. Mark price uses USDT per BTC. Open-interest units are left as
provider-native because a local units manifest was not verified. SARIMAX is
scored in its saved target units and kept separate. Any bp conversion is
conditional on the `1e6` scaling in notebook JSON cell 17, since its CSV contains
no units manifest. The Python standard-library float parser may differ from the
research script's pandas parser in approximately the 12th decimal place; results
agree to all displayed precision.

This is a local demo. No public redistribution permission has been established
for the provider-derived data, aggregates, or saved predictions. The full raw
dataset is not copied; publication still requires a provider-rights review.
