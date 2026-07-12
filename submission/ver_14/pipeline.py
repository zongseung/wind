"""Build v14 by using the pooled LightGBM for the full group-3 GBM branch.

The v14 depth backtest showed that the group-specific HistGBM dilutes the
strictly validated pooled LightGBM.  Groups 1 and 2 remain identical to v12;
group 3 uses 40% pooled LightGBM and 60% pooled MLP.  Expected-utility action,
action strength, nudges, and the mean-CF guard are inherited without tuning.
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
V12_REFERENCE_PATH = VERSION_DIR.parent / "ver_12" / "result.json"
V13_REFERENCE_PATH = VERSION_DIR.parent / "ver_13" / "result.json"
V12_SUBMISSION_PATH = VERSION_DIR.parent / "ver_12" / "submission.csv"


def reference_score(path: Path, year: int, key: str) -> float:
    result = json.loads(path.read_text())
    if path == V12_REFERENCE_PATH:
        return result["post_cv"][str(year)]["total"]["score"]
    return result["cv"][str(year)][key]["total"]["score"]


def main() -> None:
    fold_frames = {2023: V.make_2023_frames(), 2024: V.make_2024_frames()}
    fold_predictions = {}
    cv_results = {}
    cv_diagnostics = {}

    for year, frames in fold_frames.items():
        print(f"fitting v14 anchors for fold {year}")
        baseline_anchor, candidate_anchor, components = G.anchor_predictions(
            frames, pooled_gbm_only=True
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
            "v14": candidate_utility,
        }
        cv_results[str(year)] = {
            "v11_anchor": G.evaluate(frames, baseline_anchor),
            "v14_anchor": G.evaluate(frames, candidate_anchor),
            "v12_utility": G.evaluate(frames, baseline_utility),
            "v14_utility": G.evaluate(frames, candidate_utility),
            "group3_components": components,
        }
        cv_diagnostics[str(year)] = {
            "v12": baseline_diagnostics,
            "v14": candidate_diagnostics,
        }

    stores = M.robust_nudges(
        fold_frames,
        {year: values["v14"] for year, values in fold_predictions.items()},
    )
    for year, frames in fold_frames.items():
        baseline_post = M.apply_nudges(fold_predictions[year]["v12"], stores)
        candidate_post = M.apply_nudges(fold_predictions[year]["v14"], stores)
        cv_results[str(year)]["v12_post"] = G.evaluate(frames, baseline_post)
        cv_results[str(year)]["v14_post"] = G.evaluate(frames, candidate_post)
        baseline_score = cv_results[str(year)]["v12_post"]["total"]["score"]
        candidate_score = cv_results[str(year)]["v14_post"]["total"]["score"]
        cv_results[str(year)]["delta_vs_v12"] = candidate_score - baseline_score
        if V13_REFERENCE_PATH.exists():
            v13_score = reference_score(V13_REFERENCE_PATH, year, "v13_post")
            cv_results[str(year)]["delta_vs_v13"] = candidate_score - v13_score
        print(year, cv_results[str(year)])

        if V12_REFERENCE_PATH.exists():
            expected = reference_score(V12_REFERENCE_PATH, year, "")
            assert abs(baseline_score - expected) < 1e-6
        if year == 2023:
            assert abs(candidate_score - baseline_score) < 1e-12
    if V13_REFERENCE_PATH.exists():
        assert cv_results["2024"]["delta_vs_v13"] > 0.0

    print("fitting full train")
    frames = M.full_frames()
    baseline_anchor, candidate_anchor, _ = G.anchor_predictions(
        frames, pooled_gbm_only=True
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

    # The candidate changes only group 3.  Preserve the exact public-validated
    # v12 bytes for groups 1 and 2 instead of exposing grid-action ties to MPS
    # floating-point variation across otherwise identical reruns.
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
            "group3_pooled_lgbm_anchor_weight": 1.0 - M.MLP_WEIGHT,
            "group3_histgbm_anchor_weight": 0.0,
            "group3_mlp_anchor_weight": M.MLP_WEIGHT,
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
            "v14": G.mean_cf(candidate_anchor),
        },
        "test_utility_mean_cf": {
            "v12": G.mean_cf(baseline_utility),
            "v14": G.mean_cf(candidate_utility),
        },
        "test_diagnostics": {
            "v12": baseline_diagnostics,
            "v14": candidate_diagnostics,
        },
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("test mean CF", final_mean_cf)
    print(f"saved {SUBMISSION_PATH} and {RESULT_PATH}")


if __name__ == "__main__":
    main()
