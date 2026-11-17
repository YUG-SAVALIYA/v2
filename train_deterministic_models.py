"""Train deterministic Open-to-High/Open-to-Low LightGBM artifacts.

This is intentionally a *new* reproducible training recipe.  It preserves the
legacy feature definitions, targets and core LightGBM hyperparameters, but
uses one CPU thread and LightGBM's deterministic CPU settings.  Its artifacts
must not be described as the historical V2_frozen artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_hgb_all_stocks_twoday_10pct as legacy_features  # noqa: E402
import walk_forward_intraday_excursion_regression as legacy_model  # noqa: E402


CORE_PARAMETERS = {
    "objective": "regression_l1",
    "n_estimators": 450,
    "learning_rate": 0.035,
    "num_leaves": 15,
    "min_child_samples": 300,
    "reg_lambda": 12.0,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "subsample_freq": 1,
}
DETERMINISM_PARAMETERS = {
    "random_state": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "data_random_seed": 42,
    "extra_seed": 42,
    "objective_seed": 42,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbosity": -1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model() -> LGBMRegressor:
    return LGBMRegressor(**CORE_PARAMETERS, **DETERMINISM_PARAMETERS)


def fetch_ordered(con: duckdb.DuckDBPyConnection, where: str, *, sample: bool) -> pd.DataFrame:
    columns = ", ".join(["date", "symbol", "two_day_return_pct", "close_return_pct", *legacy_features.FEATURE_COLUMNS])
    sample_clause = "AND hash(symbol, date) % 20 = 0" if sample else ""
    return con.execute(
        f"SELECT {columns} FROM model_features WHERE {where} {sample_clause} ORDER BY date, symbol"
    ).fetchdf()


def model_fingerprint(model: LGBMRegressor, probe: pd.DataFrame) -> dict[str, object]:
    native_text = model.booster_.model_to_string()
    prediction = legacy_model.predict_excursion(model, probe)
    digest = hashlib.sha256()
    digest.update(np.asarray(prediction, dtype="<f8").tobytes())
    return {
        "native_model_sha256": hashlib.sha256(native_text.encode("utf-8")).hexdigest(),
        "probe_rows": int(len(probe)),
        "probe_prediction_sha256": digest.hexdigest(),
    }


def save_artifact(path: Path, model: LGBMRegressor, *, target_kind: str, train_cutoff: str, rows: int, fingerprint: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_order": list(legacy_features.FEATURE_COLUMNS),
            "target_kind": target_kind,
            "train_cutoff": train_cutoff,
            "training_rows": rows,
            "training_sample": "duckdb hash(symbol, date) % 20 = 0; ordered by date, symbol",
            "target_transform": "log1p(max(excursion_pct, 0)); inverse expm1; clip prediction at zero",
            "model_family": "lightgbm_l1_deterministic_cpu",
            "core_hyperparameters": CORE_PARAMETERS,
            "determinism_parameters": DETERMINISM_PARAMETERS,
            "fingerprint": fingerprint,
        },
        path,
        compress=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, default=ROOT / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "deterministic_models")
    parser.add_argument("--walk-forward-start", type=int, default=2020)
    parser.add_argument("--walk-forward-end", type=int, default=2026)
    parser.add_argument("--final-train-through", type=str, default=None, help="Optional YYYY-MM-DD cutoff for one post-walk-forward live model per target.")
    args = parser.parse_args()
    if args.walk_forward_start > args.walk_forward_end:
        raise ValueError("walk-forward-start must not be after walk-forward-end")
    normalized = args.normalized.resolve()
    if not normalized.exists():
        raise FileNotFoundError(f"Missing normalized OHLCV: {normalized}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    legacy_features.DATA_PATH = normalized
    con = duckdb.connect()
    created: list[dict[str, object]] = []
    for target_kind in ("intraday_high", "intraday_low"):
        con.execute("DROP TABLE IF EXISTS model_features")
        legacy_features.build_features(con, 5.0, "abs", target_kind)
        for year in range(args.walk_forward_start, args.walk_forward_end + 1):
            train = fetch_ordered(con, f"year(date) <= {year - 1}", sample=True)
            model = make_model().fit(train[legacy_features.FEATURE_COLUMNS], legacy_model.transform_target(train, target_kind))
            probe = fetch_ordered(con, f"year(date) = {year}", sample=False).head(2048)
            path = output / "walk_forward" / f"{target_kind}_{year}.joblib"
            fingerprint = model_fingerprint(model, probe)
            save_artifact(path, model, target_kind=target_kind, train_cutoff=f"{year - 1}-12-31", rows=len(train), fingerprint=fingerprint)
            created.append({"kind": "walk_forward", "target": target_kind, "test_year": year, "path": str(path.relative_to(output)), "rows": len(train), **fingerprint})
            print(f"trained deterministic {target_kind} / {year}: {len(train):,} rows")
        if args.final_train_through:
            cutoff = pd.Timestamp(args.final_train_through).date().isoformat()
            train = fetch_ordered(con, f"date <= DATE '{cutoff}'", sample=True)
            model = make_model().fit(train[legacy_features.FEATURE_COLUMNS], legacy_model.transform_target(train, target_kind))
            probe = fetch_ordered(con, f"date <= DATE '{cutoff}'", sample=False).tail(2048)
            path = output / "live" / f"{target_kind}_through_{cutoff}.joblib"
            fingerprint = model_fingerprint(model, probe)
            save_artifact(path, model, target_kind=target_kind, train_cutoff=cutoff, rows=len(train), fingerprint=fingerprint)
            created.append({"kind": "live", "target": target_kind, "training_cutoff": cutoff, "path": str(path.relative_to(output)), "rows": len(train), **fingerprint})
            print(f"trained deterministic live {target_kind} through {cutoff}: {len(train):,} rows")
    con.close()
    manifest = {
        "purpose": "Deterministic re-training recipe. Not equivalent to historical V2_frozen artifacts.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalized_sha256": sha256_file(normalized),
        "feature_order": list(legacy_features.FEATURE_COLUMNS),
        "target_definitions": {"intraday_high": "100 * (high / open - 1)", "intraday_low": "100 * (low / open - 1)"},
        "core_hyperparameters": CORE_PARAMETERS,
        "determinism_parameters": DETERMINISM_PARAMETERS,
        "sample_modulus": 20,
        "row_order": "date, symbol ascending",
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"lightgbm": lightgbm.__version__, "duckdb": duckdb.__version__, "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
        "artifacts": created,
    }
    (output / "deterministic_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(created)} deterministic artifacts to {output}")


if __name__ == "__main__":
    main()
