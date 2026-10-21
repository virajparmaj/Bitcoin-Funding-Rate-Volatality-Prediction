# Paper evidence tools

The settlement protocol and outputs remain frozen. Additive tools under
`v2/paper_tools/` evaluate narrowly scoped post-result sensitivities and produce
publication tables/figures from verified saved forecasts. These tools do not
change the primary comparison or select a more favorable result.

From repository root, using the existing v2 environment:

```bash
v2/.venv/bin/python -m pytest -q -c v2/pyproject.toml v2/tests
v2/.venv/bin/python -m v2.paper_tools.supplement --config v2/configs/paper_supplement_v1.json
v2/.venv/bin/python -m v2.paper_tools.publication.influence
v2/.venv/bin/python -m v2.paper_tools.publication.export
```

The default package is `v2/paper_package/2026-10-01/`. Its entire directory is
ignored: manuscripts, working notes, execution logs, forecasts and intermediate
outputs stay local. Configs, reusable source, tests and this guide belong in Git.
Original data and historical outputs are never overwritten. No push or submission
is part of these commands.

## Requirements and outputs

The parent study must have its ignored `panel.csv.gz` and `predictions.csv.gz`
available, in addition to the committed stage receipts. See the [study README](../README.md)
for complete acquisition and fresh-checkout reproduction. The supplement verifies
all parent receipts, original raw input hashes, numerical package versions and
frozen numerical module hashes before fitting. For a separately reproduced parent,
copy the supplement config, change its `parent`, and use a fresh `output` directory.

The supplement retains original hyperparameters and runs training-residual
mean/median calibration, 12/24-month histories and two additional RF seeds. It
saves each prediction and model/fold record before scoring. It also reports
full-main versus matched-cohort scores, a pre-evaluation funding-tail threshold,
leave-one-quarter-out summaries and approximate simultaneous conditional bounds
for the eight main model/horizon gains. All are explicitly retrospective checks.
The immutable `run/receipt.json` binds inputs, code and outputs. A completed run
can be verified again; changed code/config/output bytes are rejected.

The default influence diagnostic preserves the empty q99 evaluation tail result
and separately measures how the largest 1%/5% baseline errors affect saved scores.
Masks are paired across models and saved per event. It does not refit or change
primary samples. Its receipt records that this diagnostic was added after the
empty tail stratum was observed. This default paper diagnostic uses the original
parent and default package; adapt it explicitly for a new study version.

The publication exporter checks forecast timing, fold maturity, artifact hashes,
and rescored original/supplemental losses. It produces CSV tables and PDF/SVG/PNG
figures with `artifact_manifest.json` linking outputs to generator/input hashes.
It also writes `integrity_report.json`. Figures distinguish pointwise intervals
from simultaneous bounds. The default parent is the original completed study.

## Interpretation boundaries

The original four-hour ridge comparison remains inconclusive. One-hour RF has a
positive original secondary Holm test, but that is not a replacement primary
claim. Seed/window/cohort/error-influence results are sensitivity analyses, not
new selected discoveries. An empty tail stratum supplies no tail evidence.
Neither forecast error nor statistical significance establishes trading profit.

The local paper checklist separates code/test completion, executed experiments,
upstream data-provenance and closest-paper access gaps, and optional extensions.
A manuscript is a working draft until its authors, data rights and citations have
been reviewed. Numerical reproducibility does not establish field-level source
availability or publication priority.
