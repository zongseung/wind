"""Strict out-of-year backtest for a constrained group-3 gating network.

The two fixed v14 experts remain frozen conceptually:

- pooled metric-aligned LightGBM
- pooled three-seed MLP

A one-layer gate learns a weather-dependent LightGBM weight from blocked 2023
OOF expert predictions.  Its output is constrained to [0.25, 0.55] around the
v14 weight 0.40 and regularized toward 0.40.  The loss is a smooth version of
the exact official MAE-plus-FICR utility.  Model selection and all reported
scores use the untouched future year 2024.

This script is diagnostic and never creates a submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import wind_lib as W
import wind_pipeline as V
from submission.ver_11 import pipeline as M
from submission.ver_12 import pipeline as Q
from submission.ver_13 import pipeline as G


VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "backtest.json"
V14_RESULT_PATH = VERSION_DIR.parent / "ver_14" / "result.json"

BASE_LGB_WEIGHT = 0.40
GATE_RADIUS = 0.15
GATE_ANCHOR_PENALTY = 0.20
GATE_COEFFICIENT_PENALTY = 1e-3
SOFT_THRESHOLD_TAU = 0.005
MIN_GROUP_SCORE_DELTA = 0.0015

GATE_FEATURES = [
    "anchor_cf",
    "expert_gap_cf",
    "abs_expert_gap_cf",
    "gfs_wind_speed_100m_mean",
    "ldaps_wind_speed_10m_mean",
    "gfs_ldaps_diff",
    "gfs_ws100_grid_std",
    "ldaps_ws10_grid_std",
    "lead_h",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]


class ConstrainedGate(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.linear = nn.Linear(feature_count, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return BASE_LGB_WEIGHT + GATE_RADIUS * torch.tanh(
            self.linear(values).squeeze(-1)
        )


def blocked_2023_frames(
    period: pd.Period,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Hold one 2023 quarter out from every group to avoid same-event labels."""
    validation_start = period.start_time
    validation_end = (period + 1).start_time
    history_end = pd.Timestamp("2024-01-01")
    frames = {}
    for group in V.GROUPS:
        source = V.FR[group]
        history = source[source["kst_dtm"] < history_end]
        holdout = (history["kst_dtm"] >= validation_start) & (
            history["kst_dtm"] < validation_end
        )
        train = history.loc[~holdout]
        validation = history.loc[holdout]
        assert not train.empty and not validation.empty
        iso = W.fit_powercurve(train, V.TGT[group], W.CAP[group])
        frames[group] = (W.with_pc(train, iso), W.with_pc(validation, iso))
    return frames


def expert_predictions(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Fit both pooled experts and return group-3 predictions with gate inputs."""
    pooled_rows = []
    for group, (train, _) in frames.items():
        part = train[V.FEATS + ["kst_dtm", "stuck_mask"]].copy()
        part["cf"] = train[V.TGT[group]] / W.CAP[group]
        part["gid"] = group - 1
        part["w"] = M.mlp_weight(train)
        pooled_rows.append(part)
    pooled_train = pd.concat(pooled_rows, ignore_index=True)

    validation = frames[3][1]
    mlp_predictions = []
    for seed in M.SEEDS:
        network, scaler = V.train_one(pooled_train, seed)
        mlp_predictions.append(V.predict_one(network, scaler, validation, 3))
    mlp_prediction = np.mean(mlp_predictions, axis=0)

    pooled_lgbm = G.fit_pooled_lgbm(frames)
    lgbm_prediction = G.pooled_prediction(pooled_lgbm, validation, 3)

    output_columns = ["kst_dtm", *GATE_FEATURES[3:]]
    if V.TGT[3] in validation:
        output_columns.insert(1, V.TGT[3])
    output = validation[output_columns].copy()
    output["lgb_cf"] = lgbm_prediction / W.CAP[3]
    output["mlp_cf"] = mlp_prediction / W.CAP[3]
    if V.TGT[3] in validation:
        output["target_cf"] = validation[V.TGT[3]].to_numpy() / W.CAP[3]
    output["anchor_cf"] = (
        BASE_LGB_WEIGHT * output["lgb_cf"]
        + (1.0 - BASE_LGB_WEIGHT) * output["mlp_cf"]
    )
    output["expert_gap_cf"] = output["lgb_cf"] - output["mlp_cf"]
    output["abs_expert_gap_cf"] = output["expert_gap_cf"].abs()
    assert output[GATE_FEATURES].notna().all().all()
    return output.sort_values("kst_dtm").reset_index(drop=True)


def smooth_official_utility(
    target_cf: torch.Tensor,
    prediction_cf: torch.Tensor,
    mean_valid_cf: float,
) -> torch.Tensor:
    error = torch.sqrt((prediction_cf - target_cf) ** 2 + 1e-8)
    price = 3.0 * torch.sigmoid((0.08 - error) / SOFT_THRESHOLD_TAU)
    price += torch.sigmoid((0.06 - error) / SOFT_THRESHOLD_TAU)
    return -error + target_cf * price / (4.0 * mean_valid_cf)


def fit_gate(oof: pd.DataFrame) -> tuple[ConstrainedGate, pd.Series, pd.Series, dict]:
    mean = oof[GATE_FEATURES].mean()
    std = oof[GATE_FEATURES].std().replace(0.0, 1.0)
    values = torch.tensor(
        ((oof[GATE_FEATURES] - mean) / std).to_numpy(np.float32)
    )
    lgb_cf = torch.tensor(oof["lgb_cf"].to_numpy(np.float32))
    mlp_cf = torch.tensor(oof["mlp_cf"].to_numpy(np.float32))
    target_cf = torch.tensor(oof["target_cf"].to_numpy(np.float32))
    valid = target_cf >= W.VALID_CF
    mean_valid_cf = float(target_cf[valid].mean())

    gate = ConstrainedGate(len(GATE_FEATURES))
    optimizer = torch.optim.LBFGS(
        gate.parameters(),
        lr=0.5,
        max_iter=300,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        weights = gate(values)
        prediction = weights * lgb_cf + (1.0 - weights) * mlp_cf
        utility = smooth_official_utility(
            target_cf[valid], prediction[valid], mean_valid_cf
        ).mean()
        anchor_penalty = GATE_ANCHOR_PENALTY * (
            (weights[valid] - BASE_LGB_WEIGHT) ** 2
        ).mean()
        coefficient_penalty = GATE_COEFFICIENT_PENALTY * (
            gate.linear.weight**2
        ).mean()
        loss = -utility + anchor_penalty + coefficient_penalty
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        weights = gate(values).numpy()
    diagnostics = {
        "mean_valid_cf": mean_valid_cf,
        "training_weight_mean": float(weights.mean()),
        "training_weight_std": float(weights.std()),
        "training_weight_min": float(weights.min()),
        "training_weight_max": float(weights.max()),
        "linear_bias": float(gate.linear.bias.item()),
        "linear_coefficients": {
            feature: float(value)
            for feature, value in zip(
                GATE_FEATURES, gate.linear.weight.detach().numpy().ravel()
            )
        },
    }
    return gate, mean, std, diagnostics


def gate_weights(
    gate: ConstrainedGate,
    mean: pd.Series,
    std: pd.Series,
    frame: pd.DataFrame,
) -> np.ndarray:
    values = torch.tensor(
        ((frame[GATE_FEATURES] - mean) / std).to_numpy(np.float32)
    )
    with torch.no_grad():
        return gate(values).numpy()


def group3_quantiles(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[int, dict]:
    train, validation = frames[3]
    target = V.TGT[3]
    capacity = W.CAP[3]
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
    return {
        3: {
            "quantiles": np.sort(np.stack(predictions, axis=1), axis=1),
            "mean_valid_cf": float((train_valid[target] / capacity).mean()),
        }
    }


def score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return M.group_result(actual, prediction, W.CAP[3])


def sliced_scores(
    timestamps: pd.Series,
    actual: np.ndarray,
    baseline: np.ndarray,
    gated: np.ndarray,
    frequency: str,
) -> dict[str, dict[str, float]]:
    periods = timestamps.dt.to_period(frequency).astype(str).to_numpy()
    output = {}
    for period in sorted(np.unique(periods)):
        keep = periods == period
        base_result = score(actual[keep], baseline[keep])
        gate_result = score(actual[keep], gated[keep])
        output[period] = {
            "v14": base_result["score"],
            "gate": gate_result["score"],
            "delta": gate_result["score"] - base_result["score"],
            "one_minus_nmae_delta": (
                gate_result["one_minus_nmae"]
                - base_result["one_minus_nmae"]
            ),
            "ficr_delta": gate_result["ficr"] - base_result["ficr"],
        }
    return output


def main() -> None:
    oof_parts = []
    for period in pd.period_range("2023Q1", "2023Q4", freq="Q"):
        print("fitting OOF experts", period)
        part = expert_predictions(blocked_2023_frames(period))
        part["oof_fold"] = str(period)
        oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("kst_dtm")
    assert oof["kst_dtm"].is_unique
    gate, mean, std, gate_fit = fit_gate(oof)

    oof_weight = gate_weights(gate, mean, std, oof)
    oof_base = oof["anchor_cf"].to_numpy() * W.CAP[3]
    oof_gated = (
        oof_weight * oof["lgb_cf"].to_numpy()
        + (1.0 - oof_weight) * oof["mlp_cf"].to_numpy()
    ) * W.CAP[3]
    oof_actual = oof[V.TGT[3]].to_numpy()

    print("fitting untouched 2024 experts")
    evaluation_frames = V.make_2024_frames()
    evaluation = expert_predictions(evaluation_frames)
    weight = gate_weights(gate, mean, std, evaluation)
    base_anchor = evaluation["anchor_cf"].to_numpy() * W.CAP[3]
    gated_anchor = (
        weight * evaluation["lgb_cf"].to_numpy()
        + (1.0 - weight) * evaluation["mlp_cf"].to_numpy()
    ) * W.CAP[3]

    print("fitting group-3 quantiles")
    quantiles = group3_quantiles(evaluation_frames)
    base_utility, _ = G.apply_expected_utility({3: base_anchor}, quantiles)
    gated_utility, _ = G.apply_expected_utility({3: gated_anchor}, quantiles)
    base_prediction = base_utility[3]
    gated_prediction = gated_utility[3]
    actual = evaluation[V.TGT[3]].to_numpy()

    anchor_results = {
        "v14": score(actual, base_anchor),
        "gate": score(actual, gated_anchor),
    }
    final_results = {
        "v14": score(actual, base_prediction),
        "gate": score(actual, gated_prediction),
    }
    final_delta = final_results["gate"]["score"] - final_results["v14"]["score"]
    nmae_delta = (
        final_results["gate"]["one_minus_nmae"]
        - final_results["v14"]["one_minus_nmae"]
    )
    ficr_delta = final_results["gate"]["ficr"] - final_results["v14"]["ficr"]

    quarters = sliced_scores(
        evaluation["kst_dtm"], actual, base_prediction, gated_prediction, "Q"
    )
    months = sliced_scores(
        evaluation["kst_dtm"], actual, base_prediction, gated_prediction, "M"
    )
    quarter_deltas = np.asarray([entry["delta"] for entry in quarters.values()])
    month_deltas = np.asarray([entry["delta"] for entry in months.values()])
    weight_summary = {
        "mean": float(weight.mean()),
        "std": float(weight.std()),
        "min": float(weight.min()),
        "max": float(weight.max()),
        "p01": float(np.quantile(weight, 0.01)),
        "p50": float(np.quantile(weight, 0.50)),
        "p99": float(np.quantile(weight, 0.99)),
        "lower_boundary_fraction": float(np.mean(weight <= 0.251)),
        "upper_boundary_fraction": float(np.mean(weight >= 0.549)),
    }

    checks = {
        "material_total_gain": final_delta >= MIN_GROUP_SCORE_DELTA,
        "one_minus_nmae_nonnegative": nmae_delta >= 0.0,
        "ficr_nonnegative": ficr_delta >= 0.0,
        "quarter_mean_positive": float(quarter_deltas.mean()) > 0.0,
        "quarter_positive_fraction": float(np.mean(quarter_deltas >= 0.0)) >= 0.75,
        "month_median_positive": float(np.median(month_deltas)) > 0.0,
        "month_positive_fraction": float(np.mean(month_deltas >= 0.0)) >= 7 / 12,
        "limited_boundary_saturation": (
            weight_summary["lower_boundary_fraction"]
            + weight_summary["upper_boundary_fraction"]
        )
        <= 0.20,
    }
    decision = {
        "adopt": all(checks.values()),
        "checks": checks,
        "group_score_delta": final_delta,
        "overall_score_delta_if_other_groups_fixed": final_delta / 3.0,
        "one_minus_nmae_delta": nmae_delta,
        "ficr_delta": ficr_delta,
        "quarter_mean_delta": float(quarter_deltas.mean()),
        "quarter_positive_fraction": float(np.mean(quarter_deltas >= 0.0)),
        "month_median_delta": float(np.median(month_deltas)),
        "month_positive_fraction": float(np.mean(month_deltas >= 0.0)),
    }

    if V14_RESULT_PATH.exists():
        reference = json.loads(V14_RESULT_PATH.read_text())
        expected = reference["cv"]["2024"]["v14_post"]["groups"]["3"][
            "score"
        ]
        assert abs(final_results["v14"]["score"] - expected) < 1e-4

    result = {
        "recipe": {
            "training_year": 2023,
            "evaluation_year": 2024,
            "oof_scheme": "leave-one-2023-quarter-out across every group",
            "features": GATE_FEATURES,
            "base_lgb_weight": BASE_LGB_WEIGHT,
            "allowed_lgb_weight": [
                BASE_LGB_WEIGHT - GATE_RADIUS,
                BASE_LGB_WEIGHT + GATE_RADIUS,
            ],
            "anchor_penalty": GATE_ANCHOR_PENALTY,
            "coefficient_penalty": GATE_COEFFICIENT_PENALTY,
            "soft_threshold_tau_cf": SOFT_THRESHOLD_TAU,
            "minimum_group_score_delta": MIN_GROUP_SCORE_DELTA,
        },
        "gate_fit": gate_fit,
        "feature_mean": {key: float(value) for key, value in mean.items()},
        "feature_std": {key: float(value) for key, value in std.items()},
        "oof_training_scores": {
            "v14": score(oof_actual, oof_base),
            "gate": score(oof_actual, oof_gated),
        },
        "evaluation_anchor": anchor_results,
        "evaluation_final": final_results,
        "evaluation_weight": weight_summary,
        "quarterly": quarters,
        "monthly": months,
        "decision": decision,
        "sources": [
            "https://doi.org/10.1162/neco.1991.3.1.79",
            "https://arxiv.org/abs/2205.04216",
        ],
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("anchor", anchor_results)
    print("final", final_results)
    print("weights", weight_summary)
    print("decision", decision)
    print(f"saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
