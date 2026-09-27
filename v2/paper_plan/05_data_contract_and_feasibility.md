# Data contract and measured feasibility

## Required meanings

| Field | Meaning / rule |
|---|---|
| `funding_timestamp` | Upcoming scheduled event identifier in the original ticker; preserve in UTC |
| `funding_rate` | Current indication for that upcoming event, not automatically a settled payment |
| `predicted_funding_rate` | Provider field for the following event when available; its nullness does not remove the current indication |
| `local_timestamp` | Recorded message arrival in microseconds; conditional availability proxy |
| original `timestamp` | Synthetic in this file; only a row/provenance key, never the market clock |
| archive `calc_time` | Timestamp supplied by the official settlement archive in milliseconds; retain exactly |
| archive `last_funding_rate` | Settled-rate label candidate; verify reconciliation and units |
| `label_available_at` | Historical publication time if documented; otherwise an explicit conservative assumption with sensitivity |

The provider's [schema](https://docs.tardis.dev/downloadable-csv-files/data-types.md#derivative_ticker) supports the distinction between the upcoming indication and following-event estimate. Exact historical mapping and upstream reduction provenance remain checks, not settled assumptions.

## Measured origin coverage

The raw file spans arrival times 1 January 2020 to 17 October 2024. Its usable event IDs span **2020-01-01 08:00 UTC through 2024-10-18 00:00 UTC**. The final settlement occurs after the last recorded feature arrival; do not lose it by cutting labels at the feature date.

From the original rows, select the last nonmissing rate for the same event with arrival at or before each origin. No imputation, floor-to-hour feature matching or future-nearest selection was used.

| Lead time, hours | Age ≤30 min | Age ≤60 min | Age ≤65 min |
|---|---:|---:|---:|
| 1 | 4871 | 4931 | 5256 |
| 2 | 4877 | 4943 | 5256 |
| 4 | 4869 | 4925 | 5256 |
| 6 | 4892 | 4938 | 5256 |

All **5,256** usable event IDs have all four horizons at the 65-minute limit. Counts by event year are 1,097 / 1,095 / 1,095 / 1,095 / 874 for 2020–2024. These are potential origin records, not proof that every target or feature is valid. The 2023–2024 evaluation has at most 1,969 events before label/feature exclusions.

The difference between 60 and 65 minutes matters: boundary arrivals can leave the previous hourly observation just over 60 minutes old. Report exact ages and effective advance notice (`T_s - source_available_at`). A 65-minute tolerance does not mean observations were exactly four hours before settlement. Show 30/60-minute sensitivities on matched samples; do not optimize the tolerance for predictive scores.

## Official label feasibility probe

Read-only downloads succeeded for January 2020, June 2022 and March 2024 from the [Binance archive](https://data.binance.vision/?prefix=data/futures/um/monthly/fundingRate/BTCUSDT/). Each ZIP matched its published SHA-256 checksum; the [official repository](https://github.com/binance/binance-public-data#checksum) documents that mechanism. Stored source filenames, URLs and hashes are in [archive_probe.json](evidence/archive_probe.json).

The three archives contain 276 rows. Their header is `calc_time,funding_interval_hours,last_funding_rate`; all sampled intervals are eight hours. Match to scheduled event IDs only after retaining exact archive times. In the samples, calculation-time offsets from the scheduled hour range from 0 to 25 milliseconds. An exact timestamp join would miss some genuine candidates.

The diagnostic used a disclosed nearest-hour match within 60 seconds, required unique matches and 00/08/16 UTC alignment, and compared only ticker observations received **before** the scheduled event. This normalization is for label reconciliation, not permission to snap feature arrival times. Tighten/validate the rule over the full history and quarantine ambiguities.

| Month | Archive labels | Matched indications | Final-indication MAE, bp | Largest difference, bp | Differences >0.01 bp |
|---|---:|---:|---:|---:|---:|
| 2020-01 | 93 | 92 | 0.004457 | 0.0579 | 14 |
| 2022-06 | 90 | 90 | 0.006199 | 0.1573 | 9 |
| 2024-03 | 93 | 93 | 0.010781 | 0.3211 | 24 |

These discrepancies show why the final ticker observation is not an interchangeable settlement label. They do not show model skill. The months were selected for early/middle/stress coverage; estimates are descriptive and not representative confidence estimates.

## E0 work still required

1. Acquire all monthly label files from January 2020 through October 2024, and December 2019 for lag warm-up. The probe is not full-period coverage.
2. Verify checksums, schemas, units, interval changes, unique symbol/event mapping and duplicate conflicts. Retain exact raw timestamps and provenance.
3. Reconcile every event; report unmatched/excluded counts and label-proxy discrepancy by year. No funding targets may be forward-filled or zero-filled.
4. Preserve separate scheduled event time, calculation time and assumed publication/availability time. Use a five-minute label-publication delay as an explicitly unverified conservative default; test 0/5/15 minutes. Never use a label before its availability cutoff during fitting or feature creation.
5. Trace upstream hourly aggregation. `resample(...).last()` can combine columns from different messages. Treat row arrival as a conservative bound only conditional on the upstream process; add one-hour availability-delay sensitivity. Without original ticks, field-level point-in-time provenance remains a limitation.
6. Establish causal missing-value handling: no target imputation; train-only feature imputation with missingness/age indicators or explicit eligibility masks. No unrestricted cross-gap forward fill.

## Panel schema to implement

One row per `(venue, symbol, event_id, horizon_hours)`: scheduled settlement, actual calculation time, label-availability time, origin, source row ID, source availability, age, indication, verified settlement, residual target, eligibility reason, source checksum and protocol version. Store feature availability lineage separately for every derived group. Maintain unjoined rows in an exclusion ledger.

See [feasibility.json](evidence/feasibility.json) and [the offline probe](evidence/check_feasibility.py) for reproducible measurements. The public archive success removes the need to assume a paid ticker repull is the first blocker; it does not resolve all label/provenance issues.
