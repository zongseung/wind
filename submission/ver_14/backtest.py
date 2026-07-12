"""Backtest whether group 3 should use the pooled LightGBM for both GBM slots.

V13 conservatively combines 50% pooled LightGBM and 50% group-specific
HistGBM inside the group-3 GBM branch.  This diagnostic compares that branch
against a simpler 100% pooled LightGBM branch on the annual holdout, six
expanding quarterly folds, and monthly slices of the annual holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import wind_lib as W
import wind_pipeline as V
from submission.ver_11 import pipeline as M
from submission.ver_13 import backtest as S
from submission.ver_13 import pipeline as G


VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "backtest.json"


def branch_predictions(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    train, validation = frames[3]
    target = V.TGT[3]
    capacity = W.CAP[3]
    sample_weight = M.metric_gbm_weight(train, 3)

    histgbm = V.histgbm().fit(
        train[V.FEATS].to_numpy(),
        train[target].to_numpy(),
        sample_weight=sample_weight,
    )
    hist_prediction = histgbm.predict(validation[V.FEATS].to_numpy())
    pooled_model = G.fit_pooled_lgbm(frames)
    pooled_prediction = G.pooled_prediction(pooled_model, validation, 3)

    v13_branch = np.clip(0.5 * (pooled_prediction + hist_prediction), 0, capacity)
    pooled_only = np.clip(pooled_prediction, 0, capacity)
    actual = validation[target].to_numpy()
    diagnostics = {
        "histgbm": M.group_result(actual, hist_prediction, capacity),
        "pooled_lgbm": M.group_result(actual, pooled_prediction, capacity),
        "v13_branch": M.group_result(actual, v13_branch, capacity),
        "pooled_only": M.group_result(actual, pooled_only, capacity),
    }
    return v13_branch, pooled_only, diagnostics


def monthly_deltas(
    validation: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, dict[str, float]]:
    target = V.TGT[3]
    actual = validation[target].to_numpy()
    periods = validation["kst_dtm"].dt.to_period("M").astype(str).to_numpy()
    output = {}
    for month in sorted(np.unique(periods)):
        keep = periods == month
        base = M.group_result(actual[keep], baseline[keep], W.CAP[3])
        pool = M.group_result(actual[keep], candidate[keep], W.CAP[3])
        output[month] = {
            "v13_branch": base["score"],
            "pooled_only": pool["score"],
            "delta": pool["score"] - base["score"],
        }
    return output


def main() -> None:
    annual_frames = V.make_2024_frames()
    annual_base, annual_candidate, annual = branch_predictions(annual_frames)
    annual_delta = annual["pooled_only"]["score"] - annual["v13_branch"]["score"]
    print("annual", annual, "delta", annual_delta)

    quarterly = {}
    for period in pd.period_range("2023Q3", "2024Q4", freq="Q"):
        frames = S.cutoff_frames(period.start_time, (period + 1).start_time)
        _, _, result = branch_predictions(frames)
        delta = result["pooled_only"]["score"] - result["v13_branch"]["score"]
        quarterly[str(period)] = {
            "v13_branch": result["v13_branch"]["score"],
            "pooled_only": result["pooled_only"]["score"],
            "delta": delta,
            "one_minus_nmae_delta": (
                result["pooled_only"]["one_minus_nmae"]
                - result["v13_branch"]["one_minus_nmae"]
            ),
            "ficr_delta": result["pooled_only"]["ficr"]
            - result["v13_branch"]["ficr"],
        }
        print(period, delta)

    monthly = monthly_deltas(
        annual_frames[3][1], annual_base, annual_candidate
    )
    quarter_deltas = np.asarray([entry["delta"] for entry in quarterly.values()])
    month_deltas = np.asarray([entry["delta"] for entry in monthly.values()])
    year_means = {
        year: float(
            np.mean(
                [
                    entry["delta"]
                    for period, entry in quarterly.items()
                    if period.startswith(year)
                ]
            )
        )
        for year in ("2023", "2024")
    }
    checks = {
        "annual_positive": annual_delta > 0.0,
        "quarter_mean_positive": float(quarter_deltas.mean()) > 0.0,
        "quarter_positive_fraction": float(np.mean(quarter_deltas >= 0.0)) >= 0.60,
        "each_year_positive": min(year_means.values()) > 0.0,
        "month_median_positive": float(np.median(month_deltas)) > 0.0,
        "month_positive_fraction": float(np.mean(month_deltas >= 0.0)) >= 0.50,
    }
    decision = {
        "adopt": all(checks.values()),
        "checks": checks,
        "annual_delta": annual_delta,
        "quarter_mean_delta": float(quarter_deltas.mean()),
        "quarter_positive_fraction": float(np.mean(quarter_deltas >= 0.0)),
        "quarter_year_mean_delta": year_means,
        "month_median_delta": float(np.median(month_deltas)),
        "month_positive_fraction": float(np.mean(month_deltas >= 0.0)),
    }
    result = {
        "comparison": "v13 50% pooled LGBM + 50% specific HistGBM vs pooled LGBM only",
        "annual_2024": annual,
        "quarterly": quarterly,
        "monthly_holdout": monthly,
        "decision": decision,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("decision", decision)
    print(f"saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
