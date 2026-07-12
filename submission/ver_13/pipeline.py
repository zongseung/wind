"""Build v13 with a strictly backtested pooled LightGBM for group 3.

V12 remains unchanged for groups 1 and 2.  For group 3, the group-specific
LightGBM member is replaced by a capacity-normalized pooled LightGBM trained
on all available groups.  HistGBM and the pooled MLP keep their v12 weights,
so the new model contributes 20 percent of the group-3 point anchor.

The expected-utility action, action strength, robust nudges, and test mean-CF
guard are inherited without retuning from v12.
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
from submission.ver_12 import pipeline as Q
from submission_validation import write_submission


VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "result.json"
SUBMISSION_PATH = VERSION_DIR / "submission.csv"
BACKTEST_PATH = VERSION_DIR / "spatial_backtest.json"
REFERENCE_PATH = VERSION_DIR.parent / "ver_12" / "result.json"


def fit_pooled_lgbm(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> lgb.LGBMRegressor:
    """Fit one metric-aligned capacity-factor model across available groups."""
    rows = []
    for group, (train, _) in frames.items():
        part = train[V.FEATS].copy()
        part["group_id"] = group - 1
        part["target_cf"] = train[V.TGT[group]].to_numpy() / W.CAP[group]
        part["sample_weight"] = M.metric_gbm_weight(train, group)
        rows.append(part)
    pooled = pd.concat(rows, ignore_index=True)
    features = list(V.FEATS) + ["group_id"]
    return lgb.LGBMRegressor(**V.GBM_PARAMS).fit(
        pooled[features],
        pooled["target_cf"],
        sample_weight=pooled["sample_weight"],
    )


def pooled_prediction(
    model: lgb.LGBMRegressor,
    validation: pd.DataFrame,
    group: int,
) -> np.ndarray:
    features = list(V.FEATS) + ["group_id"]
    values = validation[V.FEATS].copy()
    values["group_id"] = group - 1
    return np.clip(model.predict(values[features]), 0, 1) * W.CAP[group]


def anchor_predictions(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    pooled_gbm_only: bool = False,
    group3_mlp_weight: float = M.MLP_WEIGHT,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Return the v11 anchor and the group-3 pooled replacement anchor."""
    pool = []
    for group, (train, _) in frames.items():
        part = train[V.FEATS + ["kst_dtm", "stuck_mask"]].copy()
        part["cf"] = train[V.TGT[group]] / W.CAP[group]
        part["gid"] = group - 1
        part["w"] = M.mlp_weight(train)
        pool.append(part)
    pooled_train = pd.concat(pool, ignore_index=True)

    mlp_predictions = {group: [] for group in frames}
    for seed in M.SEEDS:
        network, scaler = V.train_one(pooled_train, seed)
        for group, (_, validation) in frames.items():
            mlp_predictions[group].append(
                V.predict_one(network, scaler, validation, group)
            )

    pooled_lgbm = fit_pooled_lgbm(frames) if 3 in frames else None
    baseline = {}
    candidate = {}
    component_results = {}
    for group, (train, validation) in frames.items():
        capacity = W.CAP[group]
        target = V.TGT[group]
        sample_weight = M.metric_gbm_weight(train, group)
        specific_lgbm = lgb.LGBMRegressor(**V.GBM_PARAMS).fit(
            train[V.FEATS], train[target], sample_weight=sample_weight
        )
        specific_lgbm_prediction = specific_lgbm.predict(validation[V.FEATS])
        hist_prediction = (
            V.histgbm()
            .fit(
                train[V.FEATS].to_numpy(),
                train[target].to_numpy(),
                sample_weight=sample_weight,
            )
            .predict(validation[V.FEATS].to_numpy())
        )
        mlp_prediction = np.clip(
            np.mean(mlp_predictions[group], axis=0), 0, capacity
        )

        baseline_gbm = 0.5 * (specific_lgbm_prediction + hist_prediction)
        candidate_gbm = baseline_gbm
        pooled_lgbm_prediction = None
        if group == 3:
            assert pooled_lgbm is not None
            pooled_lgbm_prediction = pooled_prediction(
                pooled_lgbm, validation, group
            )
            candidate_gbm = (
                pooled_lgbm_prediction
                if pooled_gbm_only
                else 0.5 * (pooled_lgbm_prediction + hist_prediction)
            )

        baseline[group] = np.clip(
            (1.0 - M.MLP_WEIGHT) * baseline_gbm
            + M.MLP_WEIGHT * mlp_prediction,
            0,
            capacity,
        )
        candidate_mlp_weight = (
            group3_mlp_weight if group == 3 else M.MLP_WEIGHT
        )
        candidate[group] = np.clip(
            (1.0 - candidate_mlp_weight) * candidate_gbm
            + candidate_mlp_weight * mlp_prediction,
            0,
            capacity,
        )
        if group == 3 and target in validation:
            actual = validation[target].to_numpy()
            component_results = {
                "specific_lgbm": M.group_result(
                    actual, specific_lgbm_prediction, capacity
                ),
                "pooled_lgbm": M.group_result(
                    actual, pooled_lgbm_prediction, capacity
                ),
                "histgbm": M.group_result(actual, hist_prediction, capacity),
                "mlp": M.group_result(actual, mlp_prediction, capacity),
            }
    return baseline, candidate, component_results


def fit_quantile_forecasts(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[int, dict]:
    """Fit each conditional distribution once for both anchor comparisons."""
    output = {}
    for group, (train, validation) in frames.items():
        capacity = W.CAP[group]
        target = V.TGT[group]
        valid = train[target].to_numpy() >= W.VALID_CF * capacity
        train_valid = train.loc[valid]
        predictions = []
        for quantile in Q.QUANTILES:
            parameters = {**Q.QUANTILE_PARAMS, "alpha": float(quantile)}
            model = lgb.LGBMRegressor(**parameters).fit(
                train_valid[V.FEATS], train_valid[target]
            )
            predictions.append(
                np.clip(model.predict(validation[V.FEATS]), 0, capacity)
            )
        output[group] = {
            "quantiles": np.sort(np.stack(predictions, axis=1), axis=1),
            "mean_valid_cf": float((train_valid[target] / capacity).mean()),
        }
    return output


def apply_expected_utility(
    anchor: dict[int, np.ndarray],
    quantile_forecasts: dict[int, dict],
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    predictions = {}
    diagnostics = {}
    median_index = len(Q.QUANTILES) // 2
    for group, values in anchor.items():
        capacity = W.CAP[group]
        quantiles = quantile_forecasts[group]["quantiles"]
        distribution_cf = np.clip(
            (
                quantiles
                - quantiles[:, median_index, None]
                + values[:, None]
            )
            / capacity,
            0,
            1,
        )
        action = (
            Q.utility_action(
                distribution_cf, quantile_forecasts[group]["mean_valid_cf"]
            )
            * capacity
        )
        predictions[group] = np.clip(
            values + Q.ACTION_STRENGTH * (action - values), 0, capacity
        )
        diagnostics[group] = {
            "mean_action_shift_kwh": float(np.mean(action - values)),
            "mean_abs_action_shift_kwh": float(np.mean(np.abs(action - values))),
            "mean_adjusted_shift_kwh": float(
                np.mean(predictions[group] - values)
            ),
        }
    return predictions, diagnostics


def evaluate(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    predictions: dict[int, np.ndarray],
) -> dict:
    groups, total = M.evaluate_fold(frames, predictions)
    return {"groups": groups, "total": total}


def mean_cf(predictions: dict[int, np.ndarray]) -> dict[int, float]:
    return {
        group: float(100.0 * np.mean(values) / W.CAP[group])
        for group, values in predictions.items()
    }


def validate_reference(cv_results: dict) -> None:
    if not REFERENCE_PATH.exists():
        return
    reference = json.loads(REFERENCE_PATH.read_text())
    for year in (2023, 2024):
        expected = reference["post_cv"][str(year)]["total"]["score"]
        observed = cv_results[str(year)]["v12_post"]["total"]["score"]
        assert abs(observed - expected) < 1e-6, (year, observed, expected)


def main() -> None:
    fold_frames = {2023: V.make_2023_frames(), 2024: V.make_2024_frames()}
    fold_predictions = {}
    cv_results = {}
    cv_diagnostics = {}

    for year, frames in fold_frames.items():
        print(f"fitting anchors for fold {year}")
        baseline_anchor, candidate_anchor, components = anchor_predictions(frames)
        print(f"fitting quantiles for fold {year}")
        quantiles = fit_quantile_forecasts(frames)
        baseline_utility, baseline_diagnostics = apply_expected_utility(
            baseline_anchor, quantiles
        )
        candidate_utility, candidate_diagnostics = apply_expected_utility(
            candidate_anchor, quantiles
        )
        fold_predictions[year] = {
            "v12": baseline_utility,
            "v13": candidate_utility,
        }
        cv_results[str(year)] = {
            "v11_anchor": evaluate(frames, baseline_anchor),
            "v13_anchor": evaluate(frames, candidate_anchor),
            "v12_utility": evaluate(frames, baseline_utility),
            "v13_utility": evaluate(frames, candidate_utility),
            "group3_components": components,
        }
        cv_diagnostics[str(year)] = {
            "v12": baseline_diagnostics,
            "v13": candidate_diagnostics,
        }
        if REFERENCE_PATH.exists():
            reference = json.loads(REFERENCE_PATH.read_text())
            expected = reference["raw_cv"][str(year)]["v12_utility"]["total"][
                "score"
            ]
            observed = cv_results[str(year)]["v12_utility"]["total"]["score"]
            assert abs(observed - expected) < 1e-6, (year, observed, expected)
        print(year, cv_results[str(year)]["v13_utility"]["total"])

    stores = M.robust_nudges(
        fold_frames,
        {year: values["v13"] for year, values in fold_predictions.items()},
    )
    for year, frames in fold_frames.items():
        baseline_post = M.apply_nudges(fold_predictions[year]["v12"], stores)
        candidate_post = M.apply_nudges(fold_predictions[year]["v13"], stores)
        cv_results[str(year)]["v12_post"] = evaluate(frames, baseline_post)
        cv_results[str(year)]["v13_post"] = evaluate(frames, candidate_post)
        baseline_score = cv_results[str(year)]["v12_post"]["total"]["score"]
        candidate_score = cv_results[str(year)]["v13_post"]["total"]["score"]
        cv_results[str(year)]["post_delta"] = candidate_score - baseline_score
        print(f"{year} post delta", candidate_score - baseline_score)
    validate_reference(cv_results)

    print("fitting full train")
    frames = M.full_frames()
    baseline_anchor, candidate_anchor, _ = anchor_predictions(frames)
    quantiles = fit_quantile_forecasts(frames)
    baseline_utility, baseline_diagnostics = apply_expected_utility(
        baseline_anchor, quantiles
    )
    candidate_utility, candidate_diagnostics = apply_expected_utility(
        candidate_anchor, quantiles
    )
    strengths = M.guarded_nudge_strengths(candidate_utility, stores)
    test_predictions = M.apply_nudges(candidate_utility, stores, strengths)

    output = W.load_test(1)[["forecast_id", "kst_dtm"]].rename(
        columns={"kst_dtm": "forecast_kst_dtm"}
    )
    for group in V.GROUPS:
        values = test_predictions[group]
        assert np.isfinite(values).all()
        assert ((0 <= values) & (values <= W.CAP[group])).all()
        output[f"kpx_group_{group}"] = values
    assert len(output) == 8760
    write_submission(output, SUBMISSION_PATH)

    final_mean_cf = mean_cf(test_predictions)
    assert max(final_mean_cf.values()) <= M.FINAL_CF_GUARD + 1e-9
    backtest = json.loads(BACKTEST_PATH.read_text())
    result = {
        "recipe": {
            "group3_specific_lgbm_replaced_by_pooled": True,
            "pooled_lgbm_anchor_weight": (1.0 - M.MLP_WEIGHT) * 0.5,
            "histgbm_anchor_weight": (1.0 - M.MLP_WEIGHT) * 0.5,
            "mlp_anchor_weight": M.MLP_WEIGHT,
            "action_strength": Q.ACTION_STRENGTH,
            "final_cf_guard": M.FINAL_CF_GUARD,
            "postprocess_retuned": False,
        },
        "selection_backtest": {
            "path": str(BACKTEST_PATH),
            "group3_decision": backtest["decisions"]["3"],
        },
        "cv": cv_results,
        "cv_diagnostics": cv_diagnostics,
        "nudges": {str(group): list(value) for group, value in stores.items()},
        "test_nudge_strengths": strengths,
        "test_mean_cf": final_mean_cf,
        "test_anchor_mean_cf": {
            "v12": mean_cf(baseline_anchor),
            "v13": mean_cf(candidate_anchor),
        },
        "test_utility_mean_cf": {
            "v12": mean_cf(baseline_utility),
            "v13": mean_cf(candidate_utility),
        },
        "test_diagnostics": {
            "v12": baseline_diagnostics,
            "v13": candidate_diagnostics,
        },
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("test mean CF", final_mean_cf)
    print(f"saved {SUBMISSION_PATH} and {RESULT_PATH}")


if __name__ == "__main__":
    main()
