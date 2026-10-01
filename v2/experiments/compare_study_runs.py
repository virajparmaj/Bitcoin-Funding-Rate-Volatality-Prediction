"""Compare two independent full executions, excluding only destination metadata."""

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from v2.experiments.run_study import ROOT, configuration, inputs, manifest_for, save_json
from v2.src.funding_labels import sha256
from v2.src.study_store import check_store, execute_stage, execution_identity


def inspect_run(config):
    spec = configuration(config)
    _, _, archives, _ = inputs(spec)
    identity = execution_identity(manifest_for(spec, config, archives))
    output = ROOT / spec["output_dir"]
    artifacts = check_store(output, identity)
    if not (output / "_stages/report").exists():
        raise ValueError("Both independent executions must complete reporting")
    return spec, identity, output, artifacts


def compare(config, other_config):
    spec, identity, output, artifacts = inspect_run(config)
    other_spec, other_identity, other_output, other_artifacts = inspect_run(other_config)
    if output == other_output:
        raise ValueError("Reproduction must use a different output directory")
    if {k: v for k, v in spec.items() if k != "output_dir"} != {
        k: v for k, v in other_spec.items() if k != "output_dir"
    }:
        raise ValueError("Reproduction changes scientific configuration")
    for key in ["archives", "ticker_sha256", "code_sha256", "python", "packages"]:
        if identity[key] != other_identity[key]:
            raise ValueError(f"Reproduction input changed: {key}")

    def action(work):
        checks = []
        for name in sorted(artifacts):
            if name.endswith((".csv", ".csv.gz")):
                left = pd.read_csv(artifacts[name], float_precision="round_trip")
                right = pd.read_csv(other_artifacts[name], float_precision="round_trip")
                assert_frame_equal(left, right, check_exact=False, rtol=1e-9, atol=1e-12)
                checks.append(dict(artifact=name, rows=len(left), status="passed"))
        for name in [
            "selection.json",
            "baseline_selection.json",
            "coverage.json",
            "validation_candidates.json",
            "baseline_validation_candidates.json",
        ]:
            left, right = (json.loads((p / name).read_text()) for p in (output, other_output))
            if name == "selection.json":
                left, right = left["choices"], right["choices"]
            if "candidates" in name:
                assert_frame_equal(
                    pd.DataFrame(left),
                    pd.DataFrame(right),
                    check_exact=False,
                    rtol=1e-9,
                    atol=1e-12,
                )
            elif left != right:
                raise ValueError(f"Reproduction choice/validation mismatch: {name}")
            checks.append(dict(artifact=name, status="passed"))
        save_json(
            work / "reproduction_check.json",
            dict(
                status="passed",
                candidate_config=str(other_config.relative_to(ROOT)),
                candidate_manifest_sha256=sha256(other_output / "manifest.json"),
                candidate_artifacts={k: sha256(v) for k, v in other_artifacts.items()},
                checks=checks,
                rtol=1e-9,
                atol=1e-12,
                scope="Fresh full pipeline from cached sources; same machine and package environment",
            ),
        )

    def recheck():
        inspect_run(other_config)
        return inspect_run(config)[1]

    execute_stage(output, "reproduction", identity, action, recheck)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--other-config", type=Path, required=True)
    args = parser.parse_args()
    compare(args.config.resolve(), args.other_config.resolve())


if __name__ == "__main__":
    main()
