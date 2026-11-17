"""Compare frozen annual model predictions with a legacy walk-forward CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_hgb_all_stocks_twoday_10pct as legacy_features  # noqa: E402
import walk_forward_intraday_excursion_regression as legacy_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", required=True, type=Path)
    parser.add_argument("--normalized", type=Path, default=ROOT / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet")
    parser.add_argument("--models", type=Path, default=ROOT / "frozen_models")
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    legacy_features.DATA_PATH = args.normalized.resolve()
    reference = pd.read_csv(args.reference_predictions, parse_dates=["date"])
    con = duckdb.connect()
    rows = []
    for target_kind in ("intraday_low", "intraday_high"):
        con.execute("DROP TABLE IF EXISTS model_features")
        legacy_features.build_features(con, 5.0, "abs", target_kind)
        for year in range(2020, 2027):
            artifact = joblib.load(args.models / "walk_forward" / f"{target_kind}_{year}.joblib")
            test = legacy_model.fetch(con, f"year(date) = {year}", sample=False)
            predicted = legacy_model.predict_excursion(artifact["model"], test)
            expected = reference.loc[(reference.target_kind == target_kind) & (reference.test_year == year), ["date", "symbol", "predicted_excursion_pct"]].sort_values(["date", "symbol"])
            actual = pd.DataFrame({"date": pd.to_datetime(test.date), "symbol": test.symbol, "predicted": predicted}).sort_values(["date", "symbol"])
            merged = actual.merge(expected, on=["date", "symbol"], how="outer", validate="one_to_one", indicator=True)
            max_error = float(np.abs(merged.predicted - merged.predicted_excursion_pct).max()) if len(merged) else 0.0
            passed = bool((merged._merge == "both").all() and max_error <= args.atol)
            rows.append({"target_kind": target_kind, "test_year": year, "rows": len(merged), "max_abs_error": max_error, "passed": passed})
    con.close()
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    if not report.passed.all():
        raise SystemExit("Frozen-model verification failed. Check OHLCV hashes and pinned dependency versions.")
    print("Verification passed.")


if __name__ == "__main__":
    main()
