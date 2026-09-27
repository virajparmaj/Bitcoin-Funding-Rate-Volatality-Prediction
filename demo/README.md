# Funding / research demo

A local interactive presentation of the repository's original group research and later saved-result audit. Research inputs stay outside this directory and are never trained, repaired, or overwritten by the demo.

The presentation uses gold Bitcoin artwork, orange and deep-ink accents, cream surfaces, and editorial serif headings. Decorative line art sits behind the content; the charts draw only the project's exported observations and saved predictions. System fonts and local artwork keep it self-contained. See [artwork and generation prompts](ARTWORK.md) for the two decorative assets; source records and explicit evidence labels remain part of every research section.

## Run

Use Node.js 22.12+ (or a newer supported LTS) and npm. From this repository:

```sh
cd demo
npm ci
npm run dev
```

Open **http://127.0.0.1:5173/** and keep the terminal running. Use the HTTP address, not `file://` or a double-click on `index.html`: Vite compiles the TypeScript and serves the local evidence files. If port 5173 is already running this demo, reuse it; otherwise stop the conflicting process or choose `npm run dev -- --port 5174`.

For a static production build and local preview:

```sh
npm run build
npm run preview
```

Open the address printed by the preview command, usually http://127.0.0.1:4173/. `dist/` contains the resulting static site. Serve it over HTTP; no backend, model service, API key, or database is used.

## Explore

- **Overview:** dataset scope, representative RF/persistence comparison, and a 60-second summary.
- **Data & signals:** daily UTC arrival-time summaries, variable and date controls, missingness, raw examples, and searchable feature definitions.
- **Methods:** separate exploratory, classification, conditional-variance, and next-observation tasks, with targets, preprocessing, evaluation, and limitations.
- **Models & evidence:** saved RF comparisons, actual/predicted and residual charts, stored Analysis A results, unavailable classification/variance results, and a separate SARIMAX diagnostic.
- **Findings:** supported conclusions, methodology limitations, documentation corrections, and clearly marked planned work.

Source buttons open exact local excerpts and notebook output records. Hash links such as `#data`, `#methods`, `#evidence`, and `#findings` can be bookmarked. Tabs support arrow-key navigation; dialogs support Escape; chart sliders support keyboard inspection. Chart filters do not retrain a model or change the fixed original evaluation metrics.

## Rebuild the evidence

The compact exports are included in `public/data/`. To regenerate them from the repository's existing inputs, run from `demo/`:

```sh
python3 -m venv .venv
npm run export
npm run validate
```

Python 3.10+ is sufficient; the scripts use only the standard library. The command first extracts stored notebook evidence and definitions, then recomputes saved-pair arithmetic and daily summaries. It records source/output SHA-256 fingerprints in `public/data/manifest.json`. See [export details](scripts/README.md) for units, time semantics, missingness, and validation scope.

The eight evidence checks cover raw counts and missingness, source preservation, deterministic regeneration, target alignment, saved scores, units, residual pairs, daily extrema, and explicit null gaps. `npm run build` separately checks strict TypeScript and creates the static bundle. Browser interaction checks are recorded in [QA.md](QA.md).

## Evidence and release scope

Headline metrics use either **Recomputed from saved predictions** or **Stored execution output**. Narrative-only scores are confined to historical detail; unavailable scores stay unavailable. Features distinguish current definitions, source selectors, and saved-file presence. Saved RF timing is shown as row order; only raw data with a valid `local_timestamp` enter calendar charts.

This build is local and has not been published. It includes compact derived daily summaries, a small raw preview, and saved prediction pairs, not the full raw dataset. Publication clearance has not been established for these assets. The [Tardis terms](https://docs.tardis.dev/legal/terms-of-service), checked 25 September 2026, distinguish derived data from mere sampling (§1.1), condition redistribution (§9.2), and impose model/system and downstream restrictions (§9.5). A separate customer agreement may change applicable permissions (§19.1). Review the applicable account and licensor rights before any public deployment; an open-source code license is not a data license.
