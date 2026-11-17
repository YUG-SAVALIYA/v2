"""Rebuild the legacy Open-to-Low/Open-to-High historical research outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing daily per-symbol OHLCV files.")
    parser.add_argument("--recursive", action="store_true", help="Search input subdirectories too.")
    parser.add_argument("--all-files", action="store_true", help="Accept all CSV/Parquet names instead of the original *_daily_candles pattern.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse artifacts/nifty_phase1/normalized_ohlcv.parquet.")
    parser.add_argument("--skip-dashboard", action="store_true", help="Do not generate the historical HTML dashboard.")
    args = parser.parse_args()

    if not args.skip_ingest:
        sys.path.insert(0, str(ROOT / "src"))
        from nifty_research.config import DataConfig
        from nifty_research.ingest import ingest

        patterns = ["*.csv", "*.parquet", "*.pq"] if args.all_files else ["*_daily_candles.csv", "*_daily_candles.parquet", "*_daily_candles.pq"]
        config = DataConfig(
            input_path=args.input.resolve(),
            output_dir=ROOT / "artifacts" / "nifty_phase1",
            file_globs=patterns,
            recursive=args.recursive,
            duplicate_policy="last",
            extreme_move_threshold=0.30,
            exclude_uncertain_intervals=True,
            universe_membership_path=None,
        )
        result = ingest(config)
        print(f"Normalized {result.rows_written:,} rows to {result.normalized_path}")

    normalized = ROOT / "artifacts" / "nifty_phase1" / "normalized_ohlcv.parquet"
    if not normalized.exists():
        raise FileNotFoundError(f"Missing {normalized}; omit --skip-ingest or supply valid input files.")
    run([sys.executable, "scripts/walk_forward_intraday_excursion_regression.py"])
    if not args.skip_dashboard:
        run([sys.executable, "scripts/build_topn_no_stop_user_input_dashboard.py"])


if __name__ == "__main__":
    main()

