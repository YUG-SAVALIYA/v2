# V2_frozen — saved-model package

This package contains the exact 16 V2_frozen model artifacts created with the
included frozen feature code and configuration:

- 14 annual walk-forward models: High and Low for 2020–2026;
- 2 final live models fitted through the data cutoff recorded in
  `frozen_models/frozen_model_manifest.json`.

It does **not** contain market OHLCV data. Supply your own daily per-symbol
OHLCV files, then reproduce the normalised parquet input:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_legacy_pipeline.py --input D:\market_data\daily --skip-dashboard
```

For a pre-open score after the last completed candle, use the saved models; do
not retrain them:

```powershell
python score_frozen_models.py --signal-date 2026-11-16 --threshold 5 --high-cap 2 --top-n 3
```

The run records the feature order, model parameters and normalized-data SHA in
`frozen_models/frozen_model_manifest.json`. The supplied
`SHA256SUMS.txt` lets you check that no package file changed after download.

## Scope and warning

These are the **new V2_frozen snapshot artifacts**, not the unavailable
in-memory models that produced the older saved prediction CSV. Therefore do
not claim that V2_frozen exactly recreates the earlier legacy result table.
The supplied model files are the authoritative definition of V2_frozen.

This is research software only. It retains the audited limitations: no
point-in-time Nifty 500 membership, legacy missing-session handling,
unadjusted corporate-action limitations and daily-OHLC execution assumptions.
It is not approved for broker connection or live capital deployment.
