"""Build v16 with the validated constrained gate for group 3.

Gate hyperparameters are fixed by this version's backtest module.  The final gate
is fit on two leakage-safe sources available before the 2025 test period:

- blocked group-3 expert OOF predictions from 2023;
- expanding predictions for 2024 from experts trained through 2023.

The full experts then predict 2025.  Groups 1 and 2 are copied exactly from the
public-validated v12 submission; only group 3 changes from v14.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import wind_lib as W
import wind_pipeline as V
from submission.ver_11 import pipeline as M
from submission.ver_13 import pipeline as G
from submission.ver_16 import backtest as B
from submission_validation import validate_submission, write_submission


VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "result.json"
SUBMISSION_PATH = VERSION_DIR / "submission.csv"
BACKTEST_PATH = VERSION_DIR / "backtest.json"
V12_SUBMISSION_PATH = VERSION_DIR.parent / "ver_12" / "submission.csv"
V14_SUBMISSION_PATH = VERSION_DIR.parent / "ver_14" / "submission.csv"


def weight_summary(weights: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "p01": float(np.quantile(weights, 0.01)),
        "p50": float(np.quantile(weights, 0.50)),
        "p99": float(np.quantile(weights, 0.99)),
        "lower_boundary_fraction": float(np.mean(weights <= 0.251)),
        "upper_boundary_fraction": float(np.mean(weights >= 0.549)),
    }


def coefficient_vector(diagnostics: dict) -> np.ndarray:
    coefficients = diagnostics["linear_coefficients"]
    return np.asarray([coefficients[feature] for feature in B.GATE_FEATURES])


def main() -> None:
    backtest = json.loads(BACKTEST_PATH.read_text())
    if backtest["decision"]["adopt"] is not True:
        raise RuntimeError("v16 backtest did not approve the constrained gate")

    oof_parts = []
    for period in pd.period_range("2023Q1", "2023Q4", freq="Q"):
        print("fitting final-gate OOF experts", period)
        part = B.expert_predictions(B.blocked_2023_frames(period))
        part["oof_source"] = str(period)
        oof_parts.append(part)

    print("fitting expanding 2024 OOF experts")
    fold_2024 = V.make_2024_frames()
    oof_2024 = B.expert_predictions(fold_2024)
    oof_2024["oof_source"] = "2024_expanding"
    oof_parts.append(oof_2024)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values("kst_dtm")
    if oof["kst_dtm"].duplicated().any():
        raise ValueError("gate OOF timestamps must be unique")
    gate, mean, std, gate_fit = B.fit_gate(oof)
    oof_weights = B.gate_weights(gate, mean, std, oof)
    oof_base = oof["anchor_cf"].to_numpy() * W.CAP[3]
    oof_gated = (
        oof_weights * oof["lgb_cf"].to_numpy()
        + (1.0 - oof_weights) * oof["mlp_cf"].to_numpy()
    ) * W.CAP[3]
    oof_actual = oof[V.TGT[3]].to_numpy()
    oof_scores = {
        "v14_anchor": B.score(oof_actual, oof_base),
        "gate_anchor": B.score(oof_actual, oof_gated),
    }

    validated_coefficients = np.asarray(
        [
            backtest["gate_fit"]["linear_coefficients"][feature]
            for feature in B.GATE_FEATURES
        ]
    )
    final_coefficients = coefficient_vector(gate_fit)
    coefficient_cosine = float(
        np.dot(validated_coefficients, final_coefficients)
        / (
            np.linalg.norm(validated_coefficients)
            * np.linalg.norm(final_coefficients)
            + 1e-12
        )
    )
    if coefficient_cosine <= 0.25:
        raise RuntimeError("gate policy direction changed after refit")

    print("fitting full experts")
    full_frames = M.full_frames()
    test = B.expert_predictions(full_frames)
    test_weights = B.gate_weights(gate, mean, std, test)
    test_weight_stats = weight_summary(test_weights)
    boundary_fraction = (
        test_weight_stats["lower_boundary_fraction"]
        + test_weight_stats["upper_boundary_fraction"]
    )
    if boundary_fraction > 0.25:
        raise RuntimeError("test gate saturates too often")

    base_anchor = test["anchor_cf"].to_numpy() * W.CAP[3]
    gated_anchor = (
        test_weights * test["lgb_cf"].to_numpy()
        + (1.0 - test_weights) * test["mlp_cf"].to_numpy()
    ) * W.CAP[3]

    print("fitting full group-3 quantiles")
    quantiles = B.group3_quantiles(full_frames)
    gated_utility, gate_diagnostics = G.apply_expected_utility(
        {3: gated_anchor}, quantiles
    )
    group3_prediction = gated_utility[3]
    if not np.isfinite(group3_prediction).all():
        raise ValueError("group-3 prediction contains non-finite values")
    if not ((0 <= group3_prediction) & (group3_prediction <= W.CAP[3])).all():
        raise ValueError("group-3 prediction is outside the capacity bounds")
    group3_mean_cf = float(100.0 * group3_prediction.mean() / W.CAP[3])
    if group3_mean_cf > M.FINAL_CF_GUARD:
        raise ValueError("group-3 prediction violates the mean-CF guard")

    reference = pd.read_csv(V12_SUBMISSION_PATH)
    output = reference[[
        "forecast_id",
        "forecast_kst_dtm",
        "kpx_group_1",
        "kpx_group_2",
    ]].copy()
    output["kpx_group_3"] = group3_prediction
    validate_submission(output)

    v14 = pd.read_csv(V14_SUBMISSION_PATH)
    if not output[["forecast_id", "forecast_kst_dtm"]].equals(
        v14[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise ValueError("v14 and v16 submission keys do not match")
    delta = output["kpx_group_3"].to_numpy() - v14["kpx_group_3"].to_numpy()
    comparison = {
        "mean_delta_kwh": float(delta.mean()),
        "mean_abs_delta_kwh": float(np.abs(delta).mean()),
        "max_abs_delta_kwh": float(np.abs(delta).max()),
        "correlation": float(
            np.corrcoef(
                output["kpx_group_3"].to_numpy(),
                v14["kpx_group_3"].to_numpy(),
            )[0, 1]
        ),
        "v14_mean_cf": float(100.0 * v14["kpx_group_3"].mean() / W.CAP[3]),
        "v16_mean_cf": group3_mean_cf,
    }

    result = {
        "recipe": {
            "experts": ["pooled_lightgbm", "pooled_three_seed_mlp"],
            "gate_features": B.GATE_FEATURES,
            "allowed_lgb_weight": [
                B.BASE_LGB_WEIGHT - B.GATE_RADIUS,
                B.BASE_LGB_WEIGHT + B.GATE_RADIUS,
            ],
            "gate_training_sources": [
                "2023 blocked quarterly OOF",
                "2024 expanding OOF",
            ],
            "groups_1_2_exact_v12": True,
            "postprocess_retuned": False,
        },
        "selection_backtest": {
            "path": str(BACKTEST_PATH),
            "decision": backtest["decision"],
        },
        "gate_fit": gate_fit,
        "feature_mean": {key: float(value) for key, value in mean.items()},
        "feature_std": {key: float(value) for key, value in std.items()},
        "coefficient_cosine_vs_validated_gate": coefficient_cosine,
        "oof_scores": oof_scores,
        "test_weight": test_weight_stats,
        "test_expert_mean_cf": {
            "pooled_lgbm": float(100.0 * test["lgb_cf"].mean()),
            "pooled_mlp": float(100.0 * test["mlp_cf"].mean()),
            "v14_anchor": float(100.0 * base_anchor.mean() / W.CAP[3]),
            "gated_anchor": float(100.0 * gated_anchor.mean() / W.CAP[3]),
        },
        "test_group3_mean_cf": group3_mean_cf,
        "test_utility_diagnostics": gate_diagnostics[3],
        "comparison_to_v14": comparison,
        "sources": backtest["sources"],
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_submission(output, SUBMISSION_PATH)
    print("coefficient cosine", coefficient_cosine)
    print("test weights", test_weight_stats)
    print("comparison", comparison)
    print(f"saved {SUBMISSION_PATH} and {RESULT_PATH}")


if __name__ == "__main__":
    main()
