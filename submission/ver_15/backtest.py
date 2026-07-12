"""Backtest one predeclared group-3 blend change: MLP 0.60 to 0.50.

The pooled LightGBM is stronger than the pooled MLP on the only annual group-3
holdout.  To avoid a broad weight search, this script compares only v14's
40% pooled LightGBM + 60% MLP against a 50% + 50% candidate on the annual
holdout, six expanding quarterly folds, and held-out monthly slices.
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
V14_MLP_WEIGHT = 0.60
V15_MLP_WEIGHT = 0.50


def point_predictions(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    pooled_rows = []
    for group, (train, _) in frames.items():
        part = train[V.FEATS + ["kst_dtm", "stuck_mask"]].copy()
        part["cf"] = train[V.TGT[group]] / W.CAP[group]
        part["gid"] = group - 1
        part["w"] = M.mlp_weight(train)
        pooled_rows.append(part)
    pooled_train = pd.concat(pooled_rows, ignore_index=True)

    mlp_predictions = []
    validation = frames[3][1]
    for seed in M.SEEDS:
        network, scaler = V.train_one(pooled_train, seed)
        mlp_predictions.append(V.predict_one(network, scaler, validation, 3))
    mlp_prediction = np.mean(mlp_predictions, axis=0)

    pooled_lgbm = G.fit_pooled_lgbm(frames)
    lgbm_prediction = G.pooled_prediction(pooled_lgbm, validation, 3)
    v14 = np.clip(
        (1.0 - V14_MLP_WEIGHT) * lgbm_prediction
        + V14_MLP_WEIGHT * mlp_prediction,
        0,
        W.CAP[3],
    )
    v15 = np.clip(
        (1.0 - V15_MLP_WEIGHT) * lgbm_prediction
        + V15_MLP_WEIGHT * mlp_prediction,
        0,
        W.CAP[3],
    )
    actual = validation[V.TGT[3]].to_numpy()
    diagnostics = {
        "pooled_lgbm": M.group_result(actual, lgbm_prediction, W.CAP[3]),
        "mlp": M.group_result(actual, mlp_prediction, W.CAP[3]),
        "v14": M.group_result(actual, v14, W.CAP[3]),
        "v15": M.group_result(actual, v15, W.CAP[3]),
    }
    return v14, v15, diagnostics


def monthly_deltas(
    validation: pd.DataFrame,
    v14: np.ndarray,
    v15: np.ndarray,
) -> dict[str, dict[str, float]]:
    actual = validation[V.TGT[3]].to_numpy()
    periods = validation["kst_dtm"].dt.to_period("M").astype(str).to_numpy()
    output = {}
    for month in sorted(np.unique(periods)):
        keep = periods == month
        base = M.group_result(actual[keep], v14[keep], W.CAP[3])
        candidate = M.group_result(actual[keep], v15[keep], W.CAP[3])
        output[month] = {
            "v14": base["score"],
            "v15": candidate["score"],
            "delta": candidate["score"] - base["score"],
        }
    return output


def main() -> None:
    annual_frames = V.make_2024_frames()
    annual_v14, annual_v15, annual = point_predictions(annual_frames)
    annual_delta = annual["v15"]["score"] - annual["v14"]["score"]
    print("annual", annual, "delta", annual_delta)

    quarterly = {}
    for period in pd.period_range("2023Q3", "2024Q4", freq="Q"):
        frames = S.cutoff_frames(period.start_time, (period + 1).start_time)
        _, _, result = point_predictions(frames)
        delta = result["v15"]["score"] - result["v14"]["score"]
        quarterly[str(period)] = {
            "v14": result["v14"]["score"],
            "v15": result["v15"]["score"],
            "delta": delta,
            "one_minus_nmae_delta": (
                result["v15"]["one_minus_nmae"]
                - result["v14"]["one_minus_nmae"]
            ),
            "ficr_delta": result["v15"]["ficr"] - result["v14"]["ficr"],
        }
        print(period, delta)

    monthly = monthly_deltas(
        annual_frames[3][1], annual_v14, annual_v15
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
        "comparison": "v14 MLP 0.60 vs one predeclared MLP 0.50 candidate",
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
