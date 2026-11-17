"""Score a future session with frozen legacy Low/High model artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_hgb_all_stocks_twoday_10pct as legacy_features  # noqa: E402
import walk_forward_intraday_excursion_regression as legacy_model  # noqa: E402


def artifact_for(model_dir: Path, target_kind: str) -> Path:
    candidates = sorted((model_dir / "live").glob(f"{target_kind}_through_*.joblib"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one live artifact for {target_kind} in {model_dir / 'live'}; found {len(candidates)}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-date", required=True, type=pd.Timestamp, help="Next intended NSE trading-session date, YYYY-MM-DD.")
    parser.add_argument("--normalized", type=Path, default=ROOT / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet")
    parser.add_argument("--models", type=Path, default=ROOT / "frozen_models")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--high-cap", type=float, default=2.0, help="0 disables the cap.")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()
    signal_date = pd.Timestamp(args.signal_date).normalize()
    normalized = args.normalized.resolve()
    model_dir = args.models.resolve()
    if not normalized.exists():
        raise FileNotFoundError(f"Missing normalized OHLCV: {normalized}")
    manifest_path = model_dir / "frozen_model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = pd.read_parquet(normalized)
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    if signal_date <= history.date.max():
        raise ValueError("signal-date must be later than every completed OHLCV date; do not score an already completed session.")
    completed = history.loc[history.date < signal_date].copy()
    latest = completed.sort_values(["symbol", "date"], kind="stable").groupby("symbol", as_index=False).tail(1).copy()
    placeholders = latest.copy()
    placeholders["date"] = signal_date
    for column in ("open", "high", "low", "close"):
        placeholders[column] = latest["close"].to_numpy()
    placeholders["volume"] = latest["volume"].fillna(0).to_numpy()
    placeholders["is_universe_eligible"] = True
    placeholders["corporate_action_uncertain"] = False
    placeholders["is_research_eligible"] = True
    scoring_input = pd.concat([completed, placeholders], ignore_index=True)
    output = ROOT / "artifacts" / "frozen_legacy_live_scores" / signal_date.date().isoformat()
    output.mkdir(parents=True, exist_ok=True)
    temp_path = output / "_legacy_scoring_input.parquet"
    scoring_input.to_parquet(temp_path, index=False)
    legacy_features.DATA_PATH = temp_path
    con = duckdb.connect()
    predictions: list[pd.DataFrame] = []
    for target_kind in ("intraday_low", "intraday_high"):
        artifact = joblib.load(artifact_for(model_dir, target_kind))
        if artifact["feature_order"] != list(legacy_features.FEATURE_COLUMNS):
            raise RuntimeError(f"Feature-order mismatch for {target_kind}; refusing to score.")
        con.execute("DROP TABLE IF EXISTS model_features")
        legacy_features.build_features(con, 5.0, "abs", target_kind)
        score = legacy_model.fetch(con, f"date = DATE '{signal_date.date().isoformat()}'", sample=False)
        frame = score[["date", "symbol"]].copy()
        frame["target_kind"] = target_kind
        frame["predicted_excursion_pct"] = legacy_model.predict_excursion(artifact["model"], score)
        predictions.append(frame)
    con.close()
    long = pd.concat(predictions, ignore_index=True)
    wide = long.pivot(index=["date", "symbol"], columns="target_kind", values="predicted_excursion_pct").reset_index()
    wide = wide.rename(columns={"intraday_low": "predicted_low_pct", "intraday_high": "predicted_high_pct"})
    candidates = wide.loc[wide.predicted_low_pct >= args.threshold].copy()
    if args.high_cap > 0:
        candidates = candidates.loc[candidates.predicted_high_pct <= args.high_cap].copy()
    selected = candidates.sort_values(["predicted_low_pct", "symbol"], ascending=[False, True]).head(args.top_n)
    long.to_csv(output / "all_predictions_long.csv", index=False)
    wide.to_csv(output / "all_predictions.csv", index=False)
    selected.to_csv(output / "selected_signals.csv", index=False)
    (output / "score_metadata.json").write_text(json.dumps({"signal_date": str(signal_date.date()), "model_manifest": str(manifest_path), "model_data_sha256": manifest["normalized_sha256"], "threshold": args.threshold, "high_cap": args.high_cap, "top_n": args.top_n, "warning": manifest["legacy_warning"]}, indent=2), encoding="utf-8")
    print(f"Wrote {len(wide):,} predictions and {len(selected):,} selected signals to {output}")


if __name__ == "__main__":
    main()
