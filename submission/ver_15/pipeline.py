"""Build v15 with a strictly validated 50:50 group-3 point-model blend.

V14 removed the weak group-specific HistGBM.  V15 makes one predeclared
follow-up change: group 3 uses 50% pooled LightGBM and 50% pooled MLP instead
of 40% and 60%.  Groups 1 and 2, expected utility, nudges, and the mean-CF
guard remain unchanged and are not retuned.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import wind_lib as W
import wind_pipeline as V
from submission.ver_11 import pipeline as M
from submission.ver_12 import pipeline as Q
from submission.ver_13 import pipeline as G
from submission_validation import write_submission


VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "result.json"
SUBMISSION_PATH = VERSION_DIR / "submission.csv"
BACKTEST_PATH = VERSION_DIR / "backtest.json"
V12_RESULT_PATH = VERSION_DIR.parent / "ver_12" / "result.json"
V14_RESULT_PATH = VERSION_DIR.parent / "ver_14" / "result.json"
V12_SUBMISSION_PATH = VERSION_DIR.parent / "ver_12" / "submission.csv"
GROUP3_MLP_WEIGHT = 0.50
MIN_FINAL_DELTA = 1e-4


def main() -> None:
    fold_frames = {2023: V.make_2023_frames(), 2024: V.make_2024_frames()}
    fold_predictions = {}
    cv_results = {}
    cv_diagnostics = {}

    for year, frames in fold_frames.items():
        print(f"fitting v15 anchors for fold {year}")
        baseline_anchor, candidate_anchor, components = G.anchor_predictions(
            frames,
            pooled_gbm_only=True,
            group3_mlp_weight=GROUP3_MLP_WEIGHT,
        )
        print(f"fitting quantiles for fold {year}")
        quantiles = G.fit_quantile_forecasts(frames)
        baseline_utility, baseline_diagnostics = G.apply_expected_utility(
            baseline_anchor, quantiles
        )
        candidate_utility, candidate_diagnostics = G.apply_expected_utility(
            candidate_anchor, quantiles
        )
        fold_predictions[year] = {
            "v12": baseline_utility,
            "v15": candidate_utility,
        }
        cv_results[str(year)] = {
            "v11_anchor": G.evaluate(frames, baseline_anchor),
            "v15_anchor": G.evaluate(frames, candidate_anchor),
            "v12_utility": G.evaluate(frames, baseline_utility),
            "v15_utility": G.evaluate(frames, candidate_utility),
            "group3_components": components,
        }
        cv_diagnostics[str(year)] = {
            "v12": baseline_diagnostics,
            "v15": candidate_diagnostics,
        }

    stores = M.robust_nudges(
        fold_frames,
        {year: values["v15"] for year, values in fold_predictions.items()},
    )
    v12_reference = json.loads(V12_RESULT_PATH.read_text())
    v14_reference = json.loads(V14_RESULT_PATH.read_text())
    for year, frames in fold_frames.items():
        baseline_post = M.apply_nudges(fold_predictions[year]["v12"], stores)
        candidate_post = M.apply_nudges(fold_predictions[year]["v15"], stores)
        cv_results[str(year)]["v12_post"] = G.evaluate(frames, baseline_post)
        cv_results[str(year)]["v15_post"] = G.evaluate(frames, candidate_post)
        baseline_score = cv_results[str(year)]["v12_post"]["total"]["score"]
        candidate_score = cv_results[str(year)]["v15_post"]["total"]["score"]
        v14_score = v14_reference["cv"][str(year)]["v14_post"]["total"]["score"]
        cv_results[str(year)]["delta_vs_v12"] = candidate_score - baseline_score
        cv_results[str(year)]["delta_vs_v14"] = candidate_score - v14_score
        expected_v12 = v12_reference["post_cv"][str(year)]["total"]["score"]
        assert abs(baseline_score - expected_v12) < 1e-6
        if year == 2023:
            assert abs(candidate_score - baseline_score) < 1e-12
        print(year, cv_results[str(year)])
    final_delta = cv_results["2024"]["delta_vs_v14"]
    if final_delta < MIN_FINAL_DELTA:
        rejection = {
            "status": "rejected",
            "reason": "final expected-utility gain was below the adoption threshold",
            "delta_vs_v14": final_delta,
            "minimum_final_delta": MIN_FINAL_DELTA,
            "cv": cv_results,
            "cv_diagnostics": cv_diagnostics,
        }
        RESULT_PATH.write_text(
            json.dumps(rejection, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            "rejected before full fit: delta vs v14",
            final_delta,
            "<",
            MIN_FINAL_DELTA,
        )
        print(f"saved {RESULT_PATH}")
        return

    print("fitting full train")
    frames = M.full_frames()
    baseline_anchor, candidate_anchor, _ = G.anchor_predictions(
        frames,
        pooled_gbm_only=True,
        group3_mlp_weight=GROUP3_MLP_WEIGHT,
    )
    quantiles = G.fit_quantile_forecasts(frames)
    baseline_utility, baseline_diagnostics = G.apply_expected_utility(
        baseline_anchor, quantiles
    )
    candidate_utility, candidate_diagnostics = G.apply_expected_utility(
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

    reference_submission = pd.read_csv(V12_SUBMISSION_PATH)
    assert output[["forecast_id", "forecast_kst_dtm"]].equals(
        reference_submission[["forecast_id", "forecast_kst_dtm"]]
    )
    for group in (1, 2):
        output[f"kpx_group_{group}"] = reference_submission[
            f"kpx_group_{group}"
        ].to_numpy()
    write_submission(output, SUBMISSION_PATH)

    final_mean_cf = G.mean_cf(test_predictions)
    assert max(final_mean_cf.values()) <= M.FINAL_CF_GUARD + 1e-9
    backtest = json.loads(BACKTEST_PATH.read_text())
    result = {
        "recipe": {
            "group3_pooled_lgbm_anchor_weight": 1.0 - GROUP3_MLP_WEIGHT,
            "group3_histgbm_anchor_weight": 0.0,
            "group3_mlp_anchor_weight": GROUP3_MLP_WEIGHT,
            "action_strength": Q.ACTION_STRENGTH,
            "final_cf_guard": M.FINAL_CF_GUARD,
            "postprocess_retuned": False,
        },
        "selection_backtest": {
            "path": str(BACKTEST_PATH),
            "decision": backtest["decision"],
        },
        "cv": cv_results,
        "cv_diagnostics": cv_diagnostics,
        "nudges": {str(group): list(value) for group, value in stores.items()},
        "test_nudge_strengths": strengths,
        "test_mean_cf": final_mean_cf,
        "test_anchor_mean_cf": {
            "v12": G.mean_cf(baseline_anchor),
            "v15": G.mean_cf(candidate_anchor),
        },
        "test_utility_mean_cf": {
            "v12": G.mean_cf(baseline_utility),
            "v15": G.mean_cf(candidate_utility),
        },
        "test_diagnostics": {
            "v12": baseline_diagnostics,
            "v15": candidate_diagnostics,
        },
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("test mean CF", final_mean_cf)
    print(f"saved {SUBMISSION_PATH} and {RESULT_PATH}")


if __name__ == "__main__":
    main()
