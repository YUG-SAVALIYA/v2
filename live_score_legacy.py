"""Create next-session legacy hybrid predictions from completed OHLCV only.

This script is intentionally a legacy scorer.  It retrains both fixed legacy
LightGBM L1 models through the last available completed date and writes a
pre-open Low/High prediction file for the requested NSE signal date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_hgb_all_stocks_twoday_10pct as legacy_features  # noqa: E402
import walk_forward_intraday_excursion_regression as legacy_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-date", required=True, type=pd.Timestamp, help="Next intended NSE trading-session date, YYYY-MM-DD.")
    parser.add_argument("--threshold", type=float, default=5.0, help="Minimum predicted Low excursion percentage.")
    parser.add_argument("--high-cap", type=float, default=2.0, help="Maximum predicted High excursion percentage; 0 disables it.")
    parser.add_argument("--top-n", type=int, default=3, help="Maximum selected stocks.")
    args = parser.parse_args()
    signal_date = pd.Timestamp(args.signal_date).normalize()
    data_path = ROOT / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Run run_legacy_pipeline.py first; missing {data_path}")

    history = pd.read_parquet(data_path)
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    completed = history.loc[history["date"] < signal_date].copy()
    if completed.empty:
        raise ValueError("No completed OHLCV data before --signal-date.")
    latest = completed.sort_values(["symbol", "date"], kind="stable").groupby("symbol", as_index=False).tail(1).copy()
    if latest.date.max() >= signal_date:
        raise ValueError("--signal-date must be after the latest completed source session.")

    # The original feature builder emits a row's features entirely from lagged
    # history.  These valid synthetic rows provide that required signal-date
    # row without exposing its unknown OHLC values to any selected feature.
    placeholders = latest.copy()
    placeholders["date"] = signal_date
    for column in ("open", "high", "low", "close"):
        placeholders[column] = latest["close"].to_numpy()
    placeholders["volume"] = latest["volume"].fillna(0).to_numpy()
    placeholders["is_universe_eligible"] = True
    placeholders["corporate_action_uncertain"] = False
    placeholders["is_research_eligible"] = True
    scoring_input = pd.concat([completed, placeholders], ignore_index=True)

    output = ROOT / "artifacts" / "legacy_live_scores" / signal_date.date().isoformat()
    output.mkdir(parents=True, exist_ok=True)
    temp_path = output / "_legacy_scoring_input.parquet"
    scoring_input.to_parquet(temp_path, index=False)
    legacy_features.DATA_PATH = temp_path

    con = duckdb.connect()
    all_predictions: list[pd.DataFrame] = []
    cutoff = completed.date.max().date().isoformat()
    for target_kind in ("intraday_low", "intraday_high"):
        con.execute("DROP TABLE IF EXISTS model_features")
        legacy_features.build_features(con, 5.0, "abs", target_kind)
        train = legacy_model.fetch(con, f"date <= DATE '{cutoff}'", sample=True)
        score = legacy_model.fetch(con, f"date = DATE '{signal_date.date().isoformat()}'", sample=False)
        if train.empty or score.empty:
            raise RuntimeError(f"No usable {target_kind} rows for training or scoring.")
        model = legacy_model.make_lgbm().fit(
            train[legacy_features.FEATURE_COLUMNS], legacy_model.transform_target(train, target_kind)
        )
        frame = score[["date", "symbol"]].copy()
        frame["target_kind"] = target_kind
        frame["predicted_excursion_pct"] = legacy_model.predict_excursion(model, score)
        all_predictions.append(frame)
    con.close()

    predictions = pd.concat(all_predictions, ignore_index=True)
    wide = predictions.pivot(index=["date", "symbol"], columns="target_kind", values="predicted_excursion_pct").reset_index()
    wide = wide.rename(columns={"intraday_low": "predicted_low_pct", "intraday_high": "predicted_high_pct"})
    candidates = wide.loc[wide.predicted_low_pct >= args.threshold].copy()
    if args.high_cap > 0:
        candidates = candidates.loc[candidates.predicted_high_pct <= args.high_cap].copy()
    selected = candidates.sort_values(["predicted_low_pct", "symbol"], ascending=[False, True]).head(args.top_n).copy()
    predictions.to_csv(output / "all_predictions_long.csv", index=False)
    wide.to_csv(output / "all_predictions.csv", index=False)
    selected.to_csv(output / "selected_signals.csv", index=False)
    (output / "run_metadata.txt").write_text(
        f"legacy scorer\nsignal_date={signal_date.date()}\ntraining_cutoff={cutoff}\nthreshold={args.threshold}\nhigh_cap={args.high_cap}\ntop_n={args.top_n}\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(wide):,} predictions and {len(selected):,} selected signals to {output}")


if __name__ == "__main__":
    main()

