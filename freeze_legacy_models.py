"""Freeze legacy annual walk-forward and final live LightGBM model artifacts."""

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


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_hgb_all_stocks_twoday_10pct as legacy_features  # noqa: E402
import walk_forward_intraday_excursion_regression as legacy_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_artifact(path: Path, model, *, target_kind: str, train_cutoff: str, rows: int, selected_family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_order": list(legacy_features.FEATURE_COLUMNS),
            "target_kind": target_kind,
            "train_cutoff": train_cutoff,
            "training_rows": rows,
            "training_sample": "duckdb hash(symbol, date) % 20 = 0",
            "target_transform": "log1p(max(excursion_pct, 0)); inverse expm1; clip prediction at zero",
            "model_family": selected_family,
            "model_parameters": model.get_params(),
        },
        path,
        compress=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, default=ROOT / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "frozen_models")
    parser.add_argument("--include-live", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    normalized = args.normalized.resolve()
    if not normalized.exists():
        raise FileNotFoundError(f"Missing normalized OHLCV: {normalized}")
    output = args.output.resolve()
    legacy_features.DATA_PATH = normalized
    con = duckdb.connect()
    selections: dict[str, str] = {}
    created: list[dict[str, object]] = []
    for target_kind in ("intraday_high", "intraday_low"):
        con.execute("DROP TABLE IF EXISTS model_features")
        legacy_features.build_features(con, 5.0, "abs", target_kind)
        development_2018 = legacy_model.fetch(con, "year(date) <= 2018", sample=True)
        validation_2019 = legacy_model.fetch(con, "year(date) = 2019", sample=False)
        selected_name = legacy_model.select_model(development_2018, validation_2019, target_kind)
        selections[target_kind] = selected_name
        if selected_name != "lightgbm_l1":
            raise RuntimeError(f"Unexpected family {selected_name!r}; this frozen kit is defined for the documented LightGBM L1 result.")
        for year in range(2020, 2027):
            train = legacy_model.fetch(con, f"year(date) <= {year - 1}", sample=True)
            model = legacy_model.make_lgbm().fit(train[legacy_features.FEATURE_COLUMNS], legacy_model.transform_target(train, target_kind))
            path = output / "walk_forward" / f"{target_kind}_{year}.joblib"
            save_artifact(path, model, target_kind=target_kind, train_cutoff=f"{year - 1}-12-31", rows=len(train), selected_family=selected_name)
            created.append({"kind": "walk_forward", "target": target_kind, "test_year": year, "path": str(path.relative_to(output)), "rows": len(train)})
            print(f"frozen {target_kind} / {year}: {len(train):,} training rows")
        if args.include_live:
            latest = con.execute("SELECT max(date) FROM model_features").fetchone()[0]
            train = legacy_model.fetch(con, f"date <= DATE '{latest}'", sample=True)
            model = legacy_model.make_lgbm().fit(train[legacy_features.FEATURE_COLUMNS], legacy_model.transform_target(train, target_kind))
            path = output / "live" / f"{target_kind}_through_{latest}.joblib"
            save_artifact(path, model, target_kind=target_kind, train_cutoff=str(latest), rows=len(train), selected_family=selected_name)
            created.append({"kind": "live", "target": target_kind, "training_cutoff": str(latest), "path": str(path.relative_to(output)), "rows": len(train)})
            print(f"frozen live {target_kind} through {latest}: {len(train):,} training rows")
    con.close()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "legacy_warning": "Historical legacy artifact only; not production-approved. See README.md.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalized_path": str(normalized),
        "normalized_sha256": sha256(normalized),
        "feature_order": list(legacy_features.FEATURE_COLUMNS),
        "model_family_selected_from_2019_validation": selections,
        "sample_modulus": 20,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"lightgbm": lightgbm.__version__, "duckdb": duckdb.__version__, "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
        "artifacts": created,
    }
    (output / "frozen_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(created)} frozen model artifacts to {output}")


if __name__ == "__main__":
    main()
