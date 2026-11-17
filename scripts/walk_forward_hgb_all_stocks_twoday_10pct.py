"""HGB walk-forward detection of two-session +/-10% moves across all companies.

The model is deliberately pre-open: every feature is available at the close
before entry day t; target is t open to t+2 close.  Annual walk-forward folds
prevent the model or probability threshold from seeing the test year.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet"
OUTDIR = PROJECT / "deliverables" / "all_stocks_hgb_twoday_10pct_walk_forward"
POSITIVE_WEIGHT = 6.0
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
FEATURE_COLUMNS = [
    "body_lag1", "body_lag2", "body_lag3", "body_lag4", "body_lag5",
    "range_lag1", "range_lag2", "range_lag3", "range_lag4", "range_lag5",
    "return_lag1", "return_lag2", "return_lag5", "return_lag10", "return_lag20",
    "volume_ratio1", "volume_ratio5", "avg_abs_body10", "avg_range10",
    "prior_close_vs_sma5", "prior_close_vs_sma20", "prior_close_vs_high20", "prior_close_vs_low20",
    "green_count5", "red_count5", "prior_up_hit_rate20", "prior_down_hit_rate20",
    "prior_up_excursion20", "prior_down_excursion20", "avg_range20", "max_range20",
    "log_price_prior", "log_volume20", "weekday",
]


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=300,
        l2_regularization=12.0, max_iter=100, max_bins=64,
        early_stopping=False, random_state=42,
    )


def build_features(con: duckdb.DuckDBPyConnection, move_pct: float, direction: str, target_kind: str) -> str:
    source = DATA_PATH.resolve().as_posix().replace("'", "''")
    target_conditions = {
        "abs": f"abs(100.0 * (close_plus2 / open - 1)) >= {move_pct}",
        "up": f"100.0 * (close_plus2 / open - 1) >= {move_pct}",
        "down": f"100.0 * (close_plus2 / open - 1) <= -{move_pct}",
    }
    if target_kind == "two_day_close":
        target_condition = target_conditions[direction]
        outcome_expression = "100.0 * (close_plus2 / open - 1)"
        valid_target_rows = "date_plus2 IS NOT NULL AND date_diff('day', date, date_plus2) <= 7"
        target_label = f"{direction} {move_pct:g}% from entry open to close two trading sessions later"
    elif target_kind == "intraday_high":
        target_condition = f"100.0 * (high / open - 1) >= {move_pct}"
        outcome_expression = "100.0 * (high / open - 1)"
        valid_target_rows = "TRUE"
        target_label = f"intraday HIGH reaches +{move_pct:g}% from entry open (LONG target touch)"
    else:
        target_condition = f"100.0 * (low / open - 1) <= -{move_pct}"
        outcome_expression = "100.0 * (low / open - 1)"
        valid_target_rows = "TRUE"
        target_label = f"intraday LOW reaches -{move_pct:g}% from entry open (SHORT target touch)"
    con.execute(f"""
    CREATE TEMP TABLE model_features AS
    WITH raw AS (
      SELECT cast(date AS DATE) AS date, upper(symbol) AS symbol, open, high, low, close, volume
      FROM read_parquet('{source}')
      WHERE is_research_eligible AND NOT corporate_action_uncertain
    ),
    prepared AS (
      SELECT *,
        100.0 * (close / open - 1) AS body_raw,
        100.0 * (high / low - 1) AS range_raw,
        lag(close, 1) OVER w AS close_lag1,
        lag(close, 2) OVER w AS close_lag2,
        lag(close, 3) OVER w AS close_lag3,
        lag(close, 4) OVER w AS close_lag4,
        lag(close, 5) OVER w AS close_lag5,
        lag(close, 6) OVER w AS close_lag6,
        lag(close, 11) OVER w AS close_lag11,
        lag(close, 21) OVER w AS close_lag21,
        lag(open, 1) OVER w AS open_lag1,
        lag(open, 2) OVER w AS open_lag2,
        lag(open, 3) OVER w AS open_lag3,
        lag(open, 4) OVER w AS open_lag4,
        lag(open, 5) OVER w AS open_lag5,
        lag(high, 1) OVER w AS high_lag1,
        lag(high, 2) OVER w AS high_lag2,
        lag(high, 3) OVER w AS high_lag3,
        lag(high, 4) OVER w AS high_lag4,
        lag(high, 5) OVER w AS high_lag5,
        lag(low, 1) OVER w AS low_lag1,
        lag(low, 2) OVER w AS low_lag2,
        lag(low, 3) OVER w AS low_lag3,
        lag(low, 4) OVER w AS low_lag4,
        lag(low, 5) OVER w AS low_lag5,
        lag(volume, 1) OVER w AS volume_lag1,
        lead(close, 2) OVER w AS close_plus2,
        lead(date, 2) OVER w AS date_plus2,
        avg(close) OVER (w ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS sma5_prior,
        avg(close) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS sma20_prior,
        max(high) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS high20_prior,
        min(low) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS low20_prior,
        avg(volume) OVER (w ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING) AS volume20_prior,
        avg(abs(100.0 * (close / open - 1))) OVER (w ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_abs_body10,
        avg(100.0 * (high / low - 1)) OVER (w ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_range10,
        avg(CASE WHEN 100.0 * (high / open - 1) >= {move_pct} THEN 1.0 ELSE 0.0 END) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prior_up_hit_rate20,
        avg(CASE WHEN 100.0 * (low / open - 1) <= -{move_pct} THEN 1.0 ELSE 0.0 END) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prior_down_hit_rate20,
        avg(100.0 * (high / open - 1)) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prior_up_excursion20,
        avg(100.0 * (low / open - 1)) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prior_down_excursion20,
        avg(100.0 * (high / low - 1)) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS avg_range20,
        max(100.0 * (high / low - 1)) OVER (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS max_range20,
        sum(CASE WHEN close > open THEN 1 ELSE 0 END) OVER (w ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS green_count5
      FROM raw
      WINDOW w AS (PARTITION BY symbol ORDER BY date)
    )
    SELECT
      date, symbol,
      CASE WHEN date_diff('day', date, date_plus2) <= 7
           AND {target_condition} THEN 1 ELSE 0 END AS target_big_move,
      {outcome_expression} AS two_day_return_pct,
      100.0 * (close / open - 1) AS close_return_pct,
      100.0 * (close_lag1 / open_lag1 - 1) AS body_lag1,
      100.0 * (close_lag2 / open_lag2 - 1) AS body_lag2,
      100.0 * (close_lag3 / open_lag3 - 1) AS body_lag3,
      100.0 * (close_lag4 / open_lag4 - 1) AS body_lag4,
      100.0 * (close_lag5 / open_lag5 - 1) AS body_lag5,
      100.0 * (high_lag1 / low_lag1 - 1) AS range_lag1,
      100.0 * (high_lag2 / low_lag2 - 1) AS range_lag2,
      100.0 * (high_lag3 / low_lag3 - 1) AS range_lag3,
      100.0 * (high_lag4 / low_lag4 - 1) AS range_lag4,
      100.0 * (high_lag5 / low_lag5 - 1) AS range_lag5,
      100.0 * (close_lag1 / close_lag2 - 1) AS return_lag1,
      100.0 * (close_lag1 / close_lag3 - 1) AS return_lag2,
      100.0 * (close_lag1 / close_lag6 - 1) AS return_lag5,
      100.0 * (close_lag1 / close_lag11 - 1) AS return_lag10,
      100.0 * (close_lag1 / close_lag21 - 1) AS return_lag20,
      volume_lag1 / nullif(volume20_prior, 0) AS volume_ratio1,
      avg(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 6 PRECEDING AND 2 PRECEDING) / nullif(volume20_prior, 0) AS volume_ratio5,
      avg_abs_body10, avg_range10,
      100.0 * (close_lag1 / sma5_prior - 1) AS prior_close_vs_sma5,
      100.0 * (close_lag1 / sma20_prior - 1) AS prior_close_vs_sma20,
      100.0 * (close_lag1 / high20_prior - 1) AS prior_close_vs_high20,
      100.0 * (close_lag1 / low20_prior - 1) AS prior_close_vs_low20,
      green_count5, 5 - green_count5 AS red_count5,
      prior_up_hit_rate20, prior_down_hit_rate20, prior_up_excursion20, prior_down_excursion20,
      avg_range20, max_range20, ln(nullif(close_lag1, 0)) AS log_price_prior, ln(nullif(volume20_prior, 0)) AS log_volume20,
      dayofweek(date) AS weekday
    FROM prepared
    WHERE date >= DATE '2015-02-01' AND {valid_target_rows}
      AND close_lag21 IS NOT NULL AND sma20_prior IS NOT NULL AND volume20_prior IS NOT NULL
    """)
    return target_label


def fetch(con: duckdb.DuckDBPyConnection, where: str, sample: bool = False) -> pd.DataFrame:
    columns = ", ".join(["date", "symbol", "target_big_move", "two_day_return_pct", *FEATURE_COLUMNS])
    sample_clause = "AND hash(symbol, date) % 20 = 0" if sample else ""
    return con.execute(f"SELECT {columns} FROM model_features WHERE {where} {sample_clause}").fetchdf()


def threshold_table(validation: pd.DataFrame, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    y = validation.target_big_move.to_numpy(dtype=int)
    minimum_signals = max(100, int(len(validation) * 0.001))
    rows = []
    for threshold in THRESHOLDS:
        picked = probability >= threshold
        signals = int(picked.sum())
        hits = int(y[picked].sum())
        rows.append({
            "threshold": threshold, "signals": signals, "coverage_pct": round(100 * picked.mean(), 3),
            "precision_pct": round(100 * hits / signals, 2) if signals else np.nan,
            "recall_pct": round(100 * hits / y.sum(), 2) if y.sum() else np.nan,
            "eligible": signals >= minimum_signals,
        })
    table = pd.DataFrame(rows)
    candidates = table[table.eligible]
    if candidates.empty:
        candidates = table[table.signals > 0]
    selected = candidates.sort_values(["precision_pct", "recall_pct"], ascending=[False, False]).iloc[0]
    return float(selected.threshold), table


def score(test: pd.DataFrame, probability: np.ndarray, threshold: float) -> tuple[dict, pd.DataFrame]:
    y = test.target_big_move.to_numpy(dtype=int)
    picked = probability >= threshold
    hits = int(y[picked].sum())
    result = test[["date", "symbol", "target_big_move", "two_day_return_pct"]].copy()
    result["probability"] = probability
    result["selected"] = picked
    return {
        "sessions": int(len(test)), "base_event_rate_pct": round(100 * y.mean(), 3),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "pr_auc": round(float(average_precision_score(y, probability)), 4),
        "threshold": threshold, "signals": int(picked.sum()), "coverage_pct": round(100 * picked.mean(), 3),
        "precision_pct": round(100 * hits / picked.sum(), 2) if picked.any() else np.nan,
        "recall_pct": round(100 * hits / y.sum(), 2) if y.sum() else np.nan,
        "precision_lift": round((hits / picked.sum()) / y.mean(), 2) if picked.any() and y.mean() else np.nan,
    }, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward HGB detector for two-session or intraday price-move targets.")
    parser.add_argument("--move-pct", type=float, default=10.0, help="Absolute open-to-two-session-close target move, in percent.")
    parser.add_argument("--direction", choices=("abs", "up", "down"), default="abs", help="Target an absolute, upward, or downward move.")
    parser.add_argument(
        "--target-kind", choices=("two_day_close", "intraday_high", "intraday_low"), default="two_day_close",
        help="Use original two-session close target, or a same-day high/low target touch.",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    if args.move_pct <= 0:
        parser.error("--move-pct must be positive")
    direction_label = {"abs": "absolute", "up": "upward", "down": "downward"}[args.direction]
    if args.target_kind == "two_day_close":
        target_suffix = f"twoday_{args.move_pct:g}pct" + ("" if args.direction == "abs" else f"_{args.direction}")
    elif args.target_kind == "intraday_high":
        target_suffix = f"intraday_open_high_{args.move_pct:g}pct"
    else:
        target_suffix = f"intraday_open_low_{args.move_pct:g}pct"
    outdir = args.outdir or PROJECT / "deliverables" / f"all_stocks_hgb_{target_suffix}_walk_forward"
    outdir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    target_label = build_features(con, args.move_pct, args.direction, args.target_kind)
    annual_rows, threshold_rows, prediction_rows = [], [], []
    for year in range(2020, 2027):
        development = fetch(con, f"year(date) <= {year - 2}", sample=True)
        validation = fetch(con, f"year(date) = {year - 1}")
        test = fetch(con, f"year(date) = {year}")
        weight = np.where(development.target_big_move.to_numpy(dtype=bool), POSITIVE_WEIGHT, 1.0)
        threshold_model = make_model().fit(development[FEATURE_COLUMNS], development.target_big_move, sample_weight=weight)
        threshold, grid = threshold_table(validation, threshold_model.predict_proba(validation[FEATURE_COLUMNS])[:, 1])
        threshold_rows.append(grid.assign(test_year=year, selected=grid.threshold.eq(threshold)))
        train = pd.concat([development, validation], ignore_index=True)
        train_weight = np.where(train.target_big_move.to_numpy(dtype=bool), POSITIVE_WEIGHT, 1.0)
        model = make_model().fit(train[FEATURE_COLUMNS], train.target_big_move, sample_weight=train_weight)
        metrics, predictions = score(test, model.predict_proba(test[FEATURE_COLUMNS])[:, 1], threshold)
        annual_rows.append({"test_year": year, "development_sample_rows": len(development), "validation_rows": len(validation), **metrics})
        prediction_rows.append(predictions.assign(test_year=year, selected_threshold=threshold))
        print(f"{year}: precision {metrics['precision_pct']:.2f}% / {metrics['signals']} signals")
    annual = pd.DataFrame(annual_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected = predictions.selected
    aggregate = {
        "sessions": int(len(predictions)), "base_event_rate_pct": round(100 * predictions.target_big_move.mean(), 3),
        "roc_auc": round(float(roc_auc_score(predictions.target_big_move, predictions.probability)), 4),
        "pr_auc": round(float(average_precision_score(predictions.target_big_move, predictions.probability)), 4),
        "signals": int(selected.sum()), "coverage_pct": round(100 * selected.mean(), 3),
        "precision_pct": round(100 * predictions.target_big_move[selected].mean(), 2),
        "recall_pct": round(100 * predictions.target_big_move[selected].sum() / predictions.target_big_move.sum(), 2),
        "precision_lift": round(predictions.target_big_move[selected].mean() / predictions.target_big_move.mean(), 2),
    }
    annual.to_csv(outdir / "annual_metrics.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(outdir / "validation_thresholds.csv", index=False)
    predictions.to_csv(outdir / "predictions.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(outdir / "aggregate_metrics.csv", index=False)
    manifest = {"target": target_label, "features": FEATURE_COLUMNS,
                "protocol": "train through Y-2, choose threshold on Y-1, refit through Y-1, test Y", "aggregate": aggregate}
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = f"""# All-company HGB: {target_label}

Target: {target_label}.

Features: completed five-day candle bodies/ranges, returns through 20 days, volume ratios, trailing volatility, price position versus moving average/high/low, candle-colour counts and weekday.  No current-day open, gap, high, low, close or future candle is a feature.

For every test year `Y`: model development uses data through `Y-2`; probability threshold is selected only on `Y-1`; then model refits through `Y-1` and tests `Y`.

## Annual walk-forward results

{annual.to_csv(index=False)}

## Pooled out-of-sample result

{pd.DataFrame([aggregate]).to_csv(index=False)}

The selected threshold must generate at least 100 validation signals.  A >50% precision claim is only meaningful if it occurs in these annual unseen tests with non-trivial coverage.  For intraday high/low labels, daily OHLC proves that a target was touched but cannot establish the sequence of a target and any hypothetical stop-loss within that day.
"""
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(f"Wrote {outdir}")
    con.close()


if __name__ == "__main__":
    main()
