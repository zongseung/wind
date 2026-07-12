"""Build v12 by applying the corrected official-metric Bayes action to v11.

Nine valid-row LightGBM quantile forecasts approximate each conditional output
distribution.  The distribution is recentered on the stronger v11 point
forecast, then a capacity-factor grid is scored with the exact asymptotic
official utility.  Expanding-year CV fixes the action strength at 0.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import wind_lib as W
import wind_pipeline as V
from submission.ver_11 import pipeline as M
from submission_validation import write_submission


QUANTILES = np.arange(0.1, 1.0, 0.1)
ACTION_STRENGTH = 0.5
ACTION_GRID = np.linspace(0.0, 1.0, 201, dtype=np.float32)
VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "result.json"
SUBMISSION_PATH = VERSION_DIR / "submission.csv"

QUANTILE_PARAMS = dict(
    objective="quantile",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)


def utility_action(distribution_cf: np.ndarray, mean_valid_cf: float) -> np.ndarray:
    """Maximize the official MAE-plus-FICR utility on a fixed CF grid."""
    action = np.empty(len(distribution_cf), dtype=float)
    for start in range(0, len(distribution_cf), 512):
        distribution = distribution_cf[start : start + 512, None, :]
        error = np.abs(distribution - ACTION_GRID[None, :, None])
        price = np.select(
            [error <= 0.06, error <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        utility = (
            -error + distribution * price / (4.0 * mean_valid_cf)
        ).mean(axis=2)
        action[start : start + len(utility)] = ACTION_GRID[utility.argmax(axis=1)]
    return action


def expected_utility_predictions(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    anchor: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    """Fit valid-row conditional quantiles and adjust each anchor prediction."""
    predictions = {}
    diagnostics = {}
    for group, (tr, va) in frames.items():
        cap = W.CAP[group]
        target = V.TGT[group]
        valid = tr[target].to_numpy() >= W.VALID_CF * cap
        train_valid = tr.loc[valid]

        quantile_predictions = []
        for quantile in QUANTILES:
            params = {**QUANTILE_PARAMS, "alpha": float(quantile)}
            model = lgb.LGBMRegressor(**params).fit(
                train_valid[V.FEATS], train_valid[target]
            )
            quantile_predictions.append(
                np.clip(model.predict(va[V.FEATS]), 0, cap)
            )

        quantile_predictions = np.sort(
            np.stack(quantile_predictions, axis=1), axis=1
        )
        centered_distribution = np.clip(
            (
                quantile_predictions
                - quantile_predictions[:, len(QUANTILES) // 2, None]
                + anchor[group][:, None]
            )
            / cap,
            0,
            1,
        )
        mean_valid_cf = float((train_valid[target] / cap).mean())
        action = utility_action(centered_distribution, mean_valid_cf) * cap
        predictions[group] = np.clip(
            anchor[group] + ACTION_STRENGTH * (action - anchor[group]),
            0,
            cap,
        )
        diagnostics[group] = {
            "mean_action_shift_kwh": float(np.mean(action - anchor[group])),
            "mean_abs_action_shift_kwh": float(np.mean(np.abs(action - anchor[group]))),
            "mean_adjusted_shift_kwh": float(np.mean(predictions[group] - anchor[group])),
        }
    return predictions, diagnostics


def main() -> None:
    fold_frames = {2023: V.make_2023_frames(), 2024: V.make_2024_frames()}
    anchor_predictions = {}
    utility_predictions = {}
    raw_results = {}
    diagnostics = {}

    for year, frames in fold_frames.items():
        print(f"fitting v11 anchor for fold {year}")
        anchor_predictions[year] = M.predict_metric_blend(frames)
        utility_predictions[year], diagnostics[year] = expected_utility_predictions(
            frames, anchor_predictions[year]
        )
        anchor_groups, anchor_total = M.evaluate_fold(
            frames, anchor_predictions[year]
        )
        groups, total = M.evaluate_fold(frames, utility_predictions[year])
        raw_results[year] = {
            "v11_anchor": {"groups": anchor_groups, "total": anchor_total},
            "v12_utility": {"groups": groups, "total": total},
        }
        print(year, raw_results[year])

    stores = M.robust_nudges(fold_frames, utility_predictions)
    post_results = {}
    for year, frames in fold_frames.items():
        post = M.apply_nudges(utility_predictions[year], stores)
        groups, total = M.evaluate_fold(frames, post)
        post_results[year] = {"groups": groups, "total": total}
        print(f"{year} post", total)
    print("nudges", stores)

    print("fitting full train")
    frames = M.full_frames()
    test_anchor = M.predict_metric_blend(frames)
    test_utility, test_diagnostics = expected_utility_predictions(
        frames, test_anchor
    )
    strengths = M.guarded_nudge_strengths(test_utility, stores)
    test_predictions = M.apply_nudges(test_utility, stores, strengths)

    out = W.load_test(1)[["forecast_id", "kst_dtm"]].rename(
        columns={"kst_dtm": "forecast_kst_dtm"}
    )
    for group in V.GROUPS:
        values = test_predictions[group]
        assert np.isfinite(values).all()
        assert ((0 <= values) & (values <= W.CAP[group])).all()
        out[f"kpx_group_{group}"] = values
    assert len(out) == 8760
    write_submission(out, SUBMISSION_PATH)

    test_cf = {
        group: float(100.0 * np.mean(test_predictions[group]) / W.CAP[group])
        for group in V.GROUPS
    }
    assert max(test_cf.values()) <= M.FINAL_CF_GUARD + 1e-9
    result = {
        "recipe": {
            "quantiles": QUANTILES.tolist(),
            "action_strength": ACTION_STRENGTH,
            "action_grid_step_cf": float(ACTION_GRID[1] - ACTION_GRID[0]),
            "final_cf_guard": M.FINAL_CF_GUARD,
        },
        "raw_cv": raw_results,
        "post_cv": post_results,
        "cv_diagnostics": diagnostics,
        "nudges": {str(group): list(value) for group, value in stores.items()},
        "test_nudge_strengths": strengths,
        "test_diagnostics": test_diagnostics,
        "test_mean_cf": test_cf,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("test mean CF", test_cf)
    print(f"saved {SUBMISSION_PATH} and {RESULT_PATH}")


if __name__ == "__main__":
    main()
