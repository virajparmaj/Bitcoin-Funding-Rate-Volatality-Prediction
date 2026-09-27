#!/usr/bin/env python3
"""Validate local evidence exports, provenance, units, gaps, and input integrity."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import export_evidence as export


def protected_snapshot() -> dict[str, str]:
    """Hash tracked repository files outside the additive demo, not just inputs."""
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=export.ROOT).decode().split("\0")
    return {name: export.sha256(export.ROOT / name) for name in paths
            if name and not name.startswith("demo/") and (export.ROOT / name).is_file()}


class EvidenceValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = protected_snapshot()
        cls.evidence, cls.series = export.build_exports()

    @classmethod
    def tearDownClass(cls):
        if cls.before != protected_snapshot():
            raise AssertionError("Protected tracked research files changed")

    def test_raw_metadata_and_all_missingness(self):
        data = self.evidence["dataset"]
        self.assertEqual((data["row_count"], data["column_count"]), (42179, 11))
        self.assertEqual(data["arrival_start_utc"], "2020-01-01T00:59:57.141727Z")
        self.assertEqual(data["arrival_end_utc"], "2024-10-17T23:59:59.736682Z")
        self.assertEqual(data["funding_event_count"], 5256)
        self.assertEqual(data["events_with_multiple_distinct_rates"], 3664)
        self.assertEqual({c["name"]: c["missing_count"] for c in data["columns"]}, {
            "exchange": 1, "symbol": 1, "local_timestamp": 1, "funding_timestamp": 132,
            "funding_rate": 132, "predicted_funding_rate": 42179, "open_interest": 3326,
            "last_price": 3, "index_price": 5470, "mark_price": 132, "timestamp": 0})
        for field in data["columns"]:
            self.assertEqual(field["valid_count"] + field["missing_count"], 42179)
        self.assertEqual(self.series["excluded_raw_rows_missing_arrival"], 1)

    def test_saved_scores_against_independent_research_artifact(self):
        with (export.ROOT / "v2/research/saved_prediction_scores.csv").open() as handle:
            reference = {r["model"]: r for r in csv.DictReader(handle)}
        names = {"current": "Current rate reconstructed (persistence)", "predicted": "RF saved predictions",
                 "ema3": "EMA3", "lag1": "Lag1 column (one observation older)"}
        for measured in self.evidence["rf"]["metrics"]:
            expected = reference[names[measured["id"]]]
            self.assertEqual(measured["n"], int(expected["n"]))
            for key in ("r2", "mae_bp", "rmse_bp", "mse_native"):
                self.assertTrue(math.isclose(measured[key], float(expected[key]), rel_tol=1e-10, abs_tol=1e-13), (key, measured[key], expected[key]))
            for key in ("source", "task", "target", "units", "n", "evaluation", "status"):
                self.assertIn(key, measured)
            self.assertEqual(measured["status"], "Recomputed from saved predictions")
        self.assertAlmostEqual(self.evidence["rf"]["comparison"]["excess_mse_pct"], 25.615750695287876, places=8)
        self.assertAlmostEqual(self.evidence["rf"]["comparison"]["excess_mae_pct"], 49.0364359667397, places=8)

    def test_pair_scoring_arithmetic_and_units(self):
        scores = export.score([1, 2, 3], [1, 2, 2])
        self.assertEqual(scores["n"], 3)
        self.assertAlmostEqual(scores["mse_native"], 1 / 3)
        self.assertAlmostEqual(scores["r2"], .5)
        self.assertAlmostEqual(scores["mae_bp"], 10000 / 3)
        self.assertAlmostEqual(scores["rmse_bp"], math.sqrt(1 / 3) * 10000)
        with self.assertRaises(ValueError):
            export.score([], [])

    def test_sarimax_remains_separate_and_scaling_is_conditional(self):
        sar = self.evidence["sarimax"]
        self.assertEqual(sar["n"], 7622)
        self.assertAlmostEqual(sar["r2"], -4.807223264553311, places=12)
        self.assertAlmostEqual(sar["mae_saved_units"], 178.70594557404192, places=10)
        self.assertAlmostEqual(sar["mae_bp_assuming_scale"], 1.7870594557404191, places=12)
        self.assertEqual(sar["assumed_scaling_factor"], 1e6)
        self.assertNotIn("mae_bp", sar)
        self.assertIn("no units manifest", sar["scale_note"])

    def test_all_pairs_axis_residual_and_alignment(self):
        rows = self.series["rf"]
        self.assertEqual(len(rows), 7797)
        self.assertEqual([r["row_index"] for r in rows], list(range(7797)))
        for row in rows:
            self.assertNotIn("date", row)
            self.assertNotIn("timestamp", row)
            self.assertAlmostEqual(row["residual"], row["actual"] - row["predicted"], places=15)
        aligned = self.evidence["rf"]["alignment"]
        self.assertEqual(aligned["rows_with_both_raw_values"], 7793)
        self.assertLess(aligned["current_max_abs_discrepancy"], 1e-12)
        self.assertEqual(aligned["next_max_abs_discrepancy"], 0)
        for column in self.evidence["rf"]["constant_columns"].values():
            self.assertEqual(column["distinct_count"], 1)

    def test_daily_bands_retain_all_valid_arrivals_and_extremes(self):
        days = self.series["raw_daily"]
        self.assertEqual(len(days), 1752)
        self.assertEqual(sum(day["row_count"] for day in days), 42178)
        _, raw = export.read_csv(export.RAW)
        for field in ("funding_rate", "mark_price", "open_interest"):
            valid = [export.number(r[field]) for r in raw if export.microseconds(r["local_timestamp"]) is not None and export.number(r[field]) is not None]
            self.assertEqual(min(d[field]["min"] for d in days if d[field]["count"]), min(valid))
            self.assertEqual(max(d[field]["max"] for d in days if d[field]["count"]), max(valid))
            self.assertEqual(sum(d[field]["count"] for d in days), len(valid))
            for day in days:
                self.assertEqual(day[field]["count"] + day[field]["missing_count"], day["row_count"])

    def test_missing_days_and_fields_are_null_not_invented(self):
        fixture = [
            {"local_timestamp": "1577836800000000", "funding_rate": "-1", "mark_price": "7", "open_interest": ""},
            {"local_timestamp": "1577840400000000", "funding_rate": "9", "mark_price": "8", "open_interest": ""},
            {"local_timestamp": "1578009600000000", "funding_rate": "0", "mark_price": "9", "open_interest": "4"},
        ]
        days = export.daily_series(fixture)
        self.assertEqual(len(days), 3)
        self.assertEqual((days[0]["funding_rate"]["min"], days[0]["funding_rate"]["max"]), (-1, 9))
        self.assertIsNone(days[0]["open_interest"]["mean"])
        self.assertTrue(days[1]["missing_day"])
        self.assertEqual(days[1]["row_count"], 0)
        self.assertIsNone(days[1]["funding_rate"]["min"])
        self.assertIsNone(days[1]["funding_rate"]["mean"])
        self.assertEqual(days[0]["nominal_shortfall"], 22)

    def test_exports_determinism_hashes_and_write_boundary(self):
        before = export.input_hashes()
        subprocess.run([sys.executable, str(Path(export.__file__))], check=True, stdout=subprocess.DEVNULL)
        first = {p.name: p.read_bytes() for p in export.OUT.glob("*.json") if p.name in ("evidence.json", "series.json", "manifest.json", "research-content.json")}
        subprocess.run([sys.executable, str(Path(export.__file__))], check=True, stdout=subprocess.DEVNULL)
        second = {p.name: p.read_bytes() for p in export.OUT.glob("*.json") if p.name in first}
        self.assertEqual(first, second)
        self.assertEqual(before, export.input_hashes())
        manifest = json.loads(first["manifest.json"])
        self.assertEqual({path: manifest["input_sha256"][path] for path in before}, before)
        for path, expected in manifest["input_sha256"].items():
            self.assertEqual(export.sha256(export.ROOT / path), expected)
        for name, metadata in manifest["outputs"].items():
            self.assertEqual(metadata["sha256"], hashlib.sha256(first[name]).hexdigest())
        self.assertFalse(manifest["publication"]["public_release_cleared"])
        with self.assertRaises(PermissionError):
            export.write_export(export.ROOT / "data/should-not-be-created.json", b"{}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
