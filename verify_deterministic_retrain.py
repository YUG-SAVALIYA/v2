"""Compare two deterministic-training manifests from the same OHLCV snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPARE_FIELDS = ("normalized_sha256", "feature_order", "target_definitions", "core_hyperparameters", "determinism_parameters", "sample_modulus", "row_order", "versions")
ARTIFACT_FIELDS = ("kind", "target", "test_year", "training_cutoff", "rows", "native_model_sha256", "probe_rows", "probe_prediction_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="deterministic_training_manifest.json from the first run")
    parser.add_argument("--candidate", required=True, type=Path, help="deterministic_training_manifest.json from the other PC/run")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    differences: list[str] = []
    for field in COMPARE_FIELDS:
        if reference.get(field) != candidate.get(field):
            differences.append(f"manifest field differs: {field}")
    def key(row: dict) -> tuple:
        return (row.get("kind"), row.get("target"), row.get("test_year"), row.get("training_cutoff"))
    left = {key(row): row for row in reference["artifacts"]}
    right = {key(row): row for row in candidate["artifacts"]}
    if set(left) != set(right):
        differences.append("trained artifact set differs")
    for item in sorted(set(left) & set(right)):
        for field in ARTIFACT_FIELDS:
            if left[item].get(field) != right[item].get(field):
                differences.append(f"artifact {item} differs: {field}")
    if differences:
        print("REPRODUCIBILITY CHECK FAILED")
        for difference in differences:
            print("-", difference)
        raise SystemExit(1)
    print("REPRODUCIBILITY CHECK PASSED: data, configuration, native model hashes and probe predictions match exactly.")


if __name__ == "__main__":
    main()
