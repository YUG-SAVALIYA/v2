"""Leakage-safe regression of same-day open-to-high and open-to-low excursions.

This is a ranking/target-sizing study, not a target-touch classifier.  Inputs
are completed prior-session data only; the entry-day high/low is target data.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from walk_forward_hgb_all_stocks_twoday_10pct import DATA_PATH, FEATURE_COLUMNS, PROJECT, build_features


OUTDIR = PROJECT / "deliverables" / "all_stocks_intraday_excursion_regression"
SAMPLE_MODULUS = 20


def fetch(con: duckdb.DuckDBPyConnection, where: str, sample: bool) -> pd.DataFrame:
    columns = ", ".join(["date", "symbol", "two_day_return_pct", "close_return_pct", *FEATURE_COLUMNS])
    sample_clause = f"AND hash(symbol, date) % {SAMPLE_MODULUS} = 0" if sample else ""
    return con.execute(f"SELECT {columns} FROM model_features WHERE {where} {sample_clause}").fetchdf()


def make_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1", n_estimators=450, learning_rate=0.035,
        num_leaves=15, min_child_samples=300, reg_lambda=12.0,
        colsample_bytree=0.9, subsample=0.9, subsample_freq=1,
        random_state=42, n_jobs=4, verbosity=-1,
    )


def make_hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.07, max_leaf_nodes=15,
        min_samples_leaf=300, l2_regularization=12.0, max_iter=80,
        max_bins=64, early_stopping=False, random_state=42,
    )


def transform_target(frame: pd.DataFrame, target_kind: str) -> np.ndarray:
    excursion = frame.two_day_return_pct.to_numpy(dtype=float)
    if target_kind == "intraday_low":
        excursion = -excursion
    # Log target limits the influence of rare extreme/circuit observations.
    return np.log1p(np.clip(excursion, 0, None))


def predict_excursion(model, frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(np.expm1(model.predict(frame[FEATURE_COLUMNS])), 0.0)


def regression_metrics(frame: pd.DataFrame, predicted: np.ndarray, target_kind: str) -> tuple[dict, pd.DataFrame]:
    actual = np.expm1(transform_target(frame, target_kind))
    result = frame[["date", "symbol"]].copy()
    result["actual_excursion_pct"] = actual
    result["predicted_excursion_pct"] = predicted
    result["target_kind"] = target_kind
    # A close-exit proxy is intentionally reported separately from target touch.
    # Positive values mean a gross profit: long uses close/open; short uses open/close.
    result["close_exit_return_pct"] = frame.close_return_pct.to_numpy(dtype=float)
    if target_kind == "intraday_low":
        result["close_exit_return_pct"] *= -1
    top_rows = []
    for top_k in (1, 3, 5):
        selected = result.sort_values(["date", "predicted_excursion_pct"], ascending=[True, False]).groupby("date").head(top_k)
        top_rows.append({
            f"top_{top_k}_mean_actual_pct": round(float(selected.actual_excursion_pct.mean()), 3),
            f"top_{top_k}_median_actual_pct": round(float(selected.actual_excursion_pct.median()), 3),
            f"top_{top_k}_hit_5pct": round(100 * float((selected.actual_excursion_pct >= 5).mean()), 2),
            f"top_{top_k}_gross_close_profit_pct": round(100 * float((selected.close_exit_return_pct > 0).mean()), 2),
            f"top_{top_k}_mean_close_return_pct": round(float(selected.close_exit_return_pct.mean()), 3),
        })
    top_metrics = {key: value for row in top_rows for key, value in row.items()}
    correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else np.nan
    metrics = {
        "sessions": int(len(frame)),
        "mae_pct": round(float(mean_absolute_error(actual, predicted)), 3),
        "rmse_pct": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
        "correlation": round(float(correlation), 4),
        "base_mean_excursion_pct": round(float(actual.mean()), 3),
        **top_metrics,
    }
    return metrics, result


def select_model(development: pd.DataFrame, validation: pd.DataFrame, target_kind: str) -> str:
    y_dev = transform_target(development, target_kind)
    y_val = np.expm1(transform_target(validation, target_kind))
    candidates = {"lightgbm_l1": make_lgbm(), "hgb_absolute": make_hgb()}
    scores = {}
    for name, model in candidates.items():
        model.fit(development[FEATURE_COLUMNS], y_dev)
        scores[name] = mean_absolute_error(y_val, predict_excursion(model, validation))
    return min(scores, key=scores.get)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    results: list[pd.DataFrame] = []
    annual_rows: list[dict] = []
    selected_models: dict[str, str] = {}
    for target_kind in ("intraday_high", "intraday_low"):
        con.execute("DROP TABLE IF EXISTS model_features")
        build_features(con, 5.0, "abs", target_kind)
        development_2018 = fetch(con, "year(date) <= 2018", sample=True)
        validation_2019 = fetch(con, "year(date) = 2019", sample=False)
        selected_name = select_model(development_2018, validation_2019, target_kind)
        selected_models[target_kind] = selected_name
        factory = make_lgbm if selected_name == "lightgbm_l1" else make_hgb
        for year in range(2020, 2027):
            development = fetch(con, f"year(date) <= {year - 1}", sample=True)
            test = fetch(con, f"year(date) = {year}", sample=False)
            model = factory().fit(development[FEATURE_COLUMNS], transform_target(development, target_kind))
            metrics, predictions = regression_metrics(test, predict_excursion(model, test), target_kind)
            annual_rows.append({
                "target": target_kind, "test_year": year, "model": selected_name,
                "development_sample_rows": int(len(development)), **metrics,
            })
            results.append(predictions.assign(test_year=year, model=selected_name))
            print(f"{target_kind} {year}: MAE {metrics['mae_pct']:.3f}; top-1 mean {metrics['top_1_mean_actual_pct']:.3f}%")
    annual = pd.DataFrame(annual_rows)
    predictions = pd.concat(results, ignore_index=True)
    aggregate_rows = []
    for target_kind, group in predictions.groupby("target_kind"):
        actual = group.actual_excursion_pct.to_numpy()
        predicted = group.predicted_excursion_pct.to_numpy()
        row = {
            "target": target_kind,
            "mae_pct": round(float(mean_absolute_error(actual, predicted)), 3),
            "rmse_pct": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
            "correlation": round(float(np.corrcoef(actual, predicted)[0, 1]), 4),
            "base_mean_excursion_pct": round(float(actual.mean()), 3),
        }
        for top_k in (1, 3, 5):
            selected = group.sort_values(["date", "predicted_excursion_pct"], ascending=[True, False]).groupby("date").head(top_k)
            row.update({
                f"top_{top_k}_mean_actual_pct": round(float(selected.actual_excursion_pct.mean()), 3),
                f"top_{top_k}_median_actual_pct": round(float(selected.actual_excursion_pct.median()), 3),
                f"top_{top_k}_hit_5pct": round(100 * float((selected.actual_excursion_pct >= 5).mean()), 2),
                f"top_{top_k}_gross_close_profit_pct": round(100 * float((selected.close_exit_return_pct > 0).mean()), 2),
                f"top_{top_k}_mean_close_return_pct": round(float(selected.close_exit_return_pct.mean()), 3),
            })
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    annual.to_csv(OUTDIR / "annual_regression_metrics.csv", index=False)
    aggregate.to_csv(OUTDIR / "aggregate_regression_metrics.csv", index=False)
    predictions.to_csv(OUTDIR / "walk_forward_regression_predictions.csv", index=False)
    manifest = {
        "target": "same-day open-to-high and open-to-low percentage excursion", "move_pct_reference": 5,
        "features": FEATURE_COLUMNS, "model_selected_on_2019_mae": selected_models,
        "protocol": "model family selected on 2015-2018 train / 2019 validation; each 2020-2026 model trains through prior year then tests that year",
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = f"""# Intraday excursion regression

Two regressions estimate expected percentage excursion from the day's opening price: `open -> high` for LONG opportunity and `open -> low` for SHORT opportunity.  Inputs are completed daily features only.  A model family was selected separately for each target using 2019 validation MAE, then frozen for 2020--2026 expanding walk-forward testing.

## Pooled out-of-sample ranking results

{aggregate.to_csv(index=False)}

`top_k_hit_5pct` is the fraction of the model's top K ranked stocks each day that actually reached a 5% intraday excursion.  `top_k_gross_close_profit_pct` is the fraction that would be profitable if entered at the opening price and closed at that day's close (before brokerage, taxes, slippage and borrow costs); for SHORT, a lower close is a profit. `top_k_mean_close_return_pct` is that same close-exit return averaged across picks.

This close-exit result is separate from the 5% target-touch statistic. Daily OHLC does not reveal whether a profit target or stop-loss was hit first, so it cannot produce a reliable target-and-stop trade P&L. That requires intraday bars and explicit entry, target, stop and cost assumptions.
"""
    (OUTDIR / "report.md").write_text(report, encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"Wrote {OUTDIR}")
    con.close()


if __name__ == "__main__":
    main()
