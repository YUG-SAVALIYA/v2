"""Compare the saved legacy prediction CSV with newly frozen annual models."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_hgb_all_stocks_twoday_10pct as feat  # noqa: E402
import walk_forward_intraday_excursion_regression as reg  # noqa: E402


DATA = WORKSPACE / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet"
SAVED = WORKSPACE / "deliverables" / "all_stocks_intraday_excursion_regression" / "walk_forward_regression_predictions.csv"
MODELS = ROOT / "frozen_models"


def select_eval(pred: pd.DataFrame, daily: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    wide = pred.pivot(index=["date", "symbol"], columns="target_kind", values="predicted_excursion_pct").reset_index()
    wide = wide.rename(columns={"intraday_low": "low", "intraday_high": "high"})
    chosen = wide.merge(daily, on=["date", "symbol"], how="inner", validate="one_to_one")
    chosen = chosen.loc[(chosen.low >= 5.0) & (chosen.high <= 2.0)].copy()
    chosen = chosen.sort_values(["date", "low", "symbol"], ascending=[True, False, True]).groupby("date", as_index=False).head(3).copy()
    chosen["target_hit"] = chosen.actual_low_pct >= 5.0 * (1 - 1e-10)
    chosen["return_pct"] = np.where(chosen.target_hit, 5.0, chosen.close_short_return_pct)
    chosen["year"] = chosen.date.dt.year
    rows = []
    for year, group in chosen.groupby("year"):
        daily_return = group.groupby("date", sort=True).return_pct.sum() * 0.20
        equity = 100_000 * (1 + daily_return / 100).cumprod()
        drawdown = ((equity / equity.cummax() - 1) * 100).min()
        rows.append({"year": int(year), f"{label}_trades": len(group), f"{label}_target_hit_pct": 100 * group.target_hit.mean(), f"{label}_profitable_pct": 100 * (group.return_pct > 0).mean(), f"{label}_avg_trade_return_pct": group.return_pct.mean(), f"{label}_max_dd_pct": drawdown, f"{label}_portfolio_return_pct": (equity.iloc[-1] / 100_000 - 1) * 100})
    full_daily_return = chosen.groupby("date", sort=True).return_pct.sum() * 0.20
    full_equity = 100_000 * (1 + full_daily_return / 100).cumprod()
    overall = {"trades": len(chosen), "target_hit_pct": 100 * chosen.target_hit.mean(), "profitable_pct": 100 * (chosen.return_pct > 0).mean(), "avg_trade_return_pct": chosen.return_pct.mean(), "max_dd_pct": ((full_equity / full_equity.cummax() - 1) * 100).min()}
    return pd.DataFrame(rows), chosen, overall


def main() -> None:
    feat.DATA_PATH = DATA.resolve()
    con = duckdb.connect()
    new_parts = []
    for kind in ("intraday_low", "intraday_high"):
        con.execute("DROP TABLE IF EXISTS model_features")
        feat.build_features(con, 5.0, "abs", kind)
        for year in range(2020, 2027):
            test = reg.fetch(con, f"year(date) = {year}", sample=False)
            model = joblib.load(MODELS / "walk_forward" / f"{kind}_{year}.joblib")["model"]
            new_parts.append(pd.DataFrame({"date": pd.to_datetime(test.date), "symbol": test.symbol, "target_kind": kind, "predicted_excursion_pct": reg.predict_excursion(model, test)}))
    con.close()
    new = pd.concat(new_parts, ignore_index=True)
    old = pd.read_csv(SAVED, parse_dates=["date"])[["date", "symbol", "target_kind", "predicted_excursion_pct"]]
    data_path = DATA.resolve().as_posix().replace("'", "''")
    daily = duckdb.sql(f"SELECT cast(date AS DATE) AS date, upper(symbol) AS symbol, 100.0 * (1 - low / open) AS actual_low_pct, 100.0 * (1 - close / open) AS close_short_return_pct FROM read_parquet('{data_path}')").df()
    daily["date"] = pd.to_datetime(daily.date)
    old_y, old_selected, old_total = select_eval(old, daily, "old")
    new_y, new_selected, new_total = select_eval(new, daily, "new")
    report = old_y.merge(new_y, on="year", how="outer").sort_values("year")
    for metric in ("trades", "target_hit_pct", "profitable_pct", "avg_trade_return_pct", "max_dd_pct", "portfolio_return_pct"):
        report[f"delta_{metric}"] = report[f"new_{metric}"] - report[f"old_{metric}"]
    overlap = len(old_selected[["date", "symbol"]].merge(new_selected[["date", "symbol"]], on=["date", "symbol"]))
    print("SETTINGS: Low >= 5%, High <= 2%, Top-3 by Low, target 5%, no stop, 20% allocation")
    print("OLD_TOTAL", old_total)
    print("NEW_TOTAL", new_total)
    print(f"SELECTED_OVERLAP={overlap}; old={len(old_selected)}; new={len(new_selected)}; overlap_pct_of_old={100 * overlap / len(old_selected):.2f}")
    print(report.round(3).to_string(index=False))
    report.to_csv(ROOT / "comparison_saved_vs_frozen_yearwise.csv", index=False)


if __name__ == "__main__":
    main()
