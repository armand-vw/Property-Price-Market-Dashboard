"""
scripts/benchmark_real_data.py
==============================
Train the project's pipeline on **real** Ames Housing sales (5-fold CV) and
write a benchmark report to ``reports/``.

Usage
-----
    python scripts/benchmark_real_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from property_insights import real_data  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _row(metrics: dict, name: str) -> str:
    mae = metrics.loc["mae"]
    rmse = metrics.loc["rmse"]
    mape = metrics.loc["mape"]
    r2 = metrics.loc["r2"]
    return (
        f"| {name} | {_money(mae['mean'])} ± {_money(mae['std'])} "
        f"| {_money(rmse['mean'])} ± {_money(rmse['std'])} "
        f"| {mape['mean']:.1f}% ± {mape['std']:.1f}% "
        f"| {r2['mean']:.3f} ± {r2['std']:.3f} |"
    )


def build_report(result: dict) -> str:
    """Render the markdown benchmark report."""
    importance = result["importance"].head(10)
    importance_rows = "\n".join(
        f"| {row.feature} | {row.importance:.3f} |" for row in importance.itertuples()
    )
    best = ", ".join(f"{key.replace('regressor__model__', '')}={value}" for key, value in result["best_params"].items())

    return f"""# Real-data benchmark — Ames Housing

Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · `scripts/benchmark_real_data.py`.

## What this is

The shipped dashboard is trained on **synthetic** listings calibrated to real
Zillow medians. To give an honest, real-data read on modelling quality, the
**same leak-free pipeline** is trained and evaluated here on the **real** Ames
Housing dataset ({result['n_records']:,} actual sales, {result['n_neighborhoods']}
neighbourhoods), with 5-fold cross-validation.

- Target: `log1p(price)` (inverted to dollars for scoring).
- Preprocessing (imputation, scaling, one-hot) is fitted **inside** each fold.
- Metrics are mean ± std across {result['cv']} folds.

## Results

| Model | MAE | RMSE | MAPE | R² |
| --- | --- | --- | --- | --- |
{_row(result['baseline'], 'Baseline (predict median)')}
{_row(result['model'], 'XGBoost (project defaults)')}
{_row(result['tuned'], 'XGBoost (lightly tuned)')}

Best hyper-parameters: `{best}`.

## Top drivers of price (real data)

| Feature | Importance |
| --- | --- |
{importance_rows}

## Caveats

- Ames is a **single, static city** dataset (2006–2010 sales); it validates the
  modelling approach, not live US market coverage.
- The dashboard's synthetic listings are still used for the interactive product
  (documented in `MODEL_CARD.md`); this report is the real-data counterpart.
"""


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = real_data.load_real_data()
    result = real_data.benchmark(frame)

    report = build_report(result)
    (REPORTS_DIR / "real_data_benchmark.md").write_text(report, encoding="utf-8")

    serialisable = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_records": result["n_records"],
        "n_neighborhoods": result["n_neighborhoods"],
        "cv": result["cv"],
        "best_params": result["best_params"],
        "baseline": result["baseline"].round(4).to_dict(),
        "model": result["model"].round(4).to_dict(),
        "tuned": result["tuned"].round(4).to_dict(),
        "importance": result["importance"].round(4).to_dict(orient="records"),
    }
    (REPORTS_DIR / "real_data_benchmark.json").write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8"
    )

    print("=" * 62)
    print("Real-data benchmark complete (Ames Housing)")
    print("=" * 62)
    for name, key in [("Baseline", "baseline"), ("Model", "model"), ("Tuned", "tuned")]:
        m = result[key]
        print(
            f"{name:<9} MAE {_money(m.loc['mae']['mean']):>10}   "
            f"RMSE {_money(m.loc['rmse']['mean']):>10}   "
            f"MAPE {m.loc['mape']['mean']:5.1f}%   R² {m.loc['r2']['mean']:.3f}"
        )
    print("-" * 62)
    print(f"Report : {REPORTS_DIR / 'real_data_benchmark.md'}")


if __name__ == "__main__":
    main()
