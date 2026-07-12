"""Build a metric-aligned v11 candidate.

The official metric ignores rows whose realized capacity factor is below 10%.
V11 therefore gives those rows zero weight for the GBM members.  The pooled
MLP keeps the calibrated v7 SCADA weighting, which was more stable in the
2023 fold.  Expanding-year CV selects the blend and only permits a nudge that
does not hurt either observed fold; group 3 has one fold and is left raw.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import wind_lib as W
import wind_pipeline as V
from official_metric import group_scores
from submission_validation import write_submission


SEEDS = [15, 0, 1]
MLP_WEIGHT = 0.60
INVALID_WEIGHT = 0.0
FINAL_CF_GUARD = 39.5
VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "result.json"
SUBMISSION_PATH = VERSION_DIR / "submission.csv"


def mlp_weight(tr: pd.DataFrame) -> np.ndarray:
    """Preserve the v7 label-cleaning recipe for the stable MLP member."""
    return np.where(tr["stuck_mask"].to_numpy(), V.STUCK_W, 1.0)


def metric_gbm_weight(tr: pd.DataFrame, group: int) -> np.ndarray:
    """Match GBM empirical risk to the metric's realized-validity mask."""
    valid = tr[V.TGT[group]].to_numpy() >= W.VALID_CF * W.CAP[group]
    valid_weight = np.where(tr["stuck_mask"].to_numpy(), V.STUCK_W, 1.0)
    return np.where(valid, valid_weight, INVALID_WEIGHT)


def predict_metric_blend(
    tr_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    seeds: list[int] = SEEDS,
) -> dict[int, np.ndarray]:
    """Fit the v7 MLP plus valid-row GBMs and return their fixed blend."""
    pool = []
    for group, (tr, _) in tr_frames.items():
        p = tr[V.FEATS + ["kst_dtm", "stuck_mask"]].copy()
        p["cf"] = tr[V.TGT[group]] / W.CAP[group]
        p["gid"] = group - 1
        p["w"] = mlp_weight(tr)
        pool.append(p)
    pool_tr = pd.concat(pool, ignore_index=True)

    mlp_predictions = {group: [] for group in tr_frames}
    for seed in seeds:
        net, scaler = V.train_one(pool_tr, seed)
        for group, (_, va) in tr_frames.items():
            mlp_predictions[group].append(V.predict_one(net, scaler, va, group))

    predictions = {}
    for group, (tr, va) in tr_frames.items():
        cap = W.CAP[group]
        target = V.TGT[group]
        weight = metric_gbm_weight(tr, group)
        lgbm = lgb.LGBMRegressor(**V.GBM_PARAMS).fit(
            tr[V.FEATS], tr[target], sample_weight=weight
        )
        hgb = V.histgbm().fit(
            tr[V.FEATS].to_numpy(), tr[target].to_numpy(), sample_weight=weight
        )
        gbm_pred = 0.5 * (
            lgbm.predict(va[V.FEATS]) + hgb.predict(va[V.FEATS].to_numpy())
        )
        mlp_pred = np.mean(mlp_predictions[group], axis=0)
        predictions[group] = np.clip(
            (1.0 - MLP_WEIGHT) * gbm_pred + MLP_WEIGHT * mlp_pred,
            0,
            cap,
        )
    return predictions


def group_result(actual: np.ndarray, pred: np.ndarray, cap: int) -> dict[str, float]:
    nmae, ficr = group_scores(actual, pred, cap)
    return {
        "score": float(0.5 * (1.0 - nmae) + 0.5 * ficr),
        "one_minus_nmae": float(1.0 - nmae),
        "ficr": float(ficr),
        "mean_cf": float(100.0 * np.mean(pred) / cap),
    }


def evaluate_fold(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    pred: dict[int, np.ndarray],
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    by_group = {}
    nmae, ficr = [], []
    for group, (_, va) in frames.items():
        result = group_result(va[V.TGT[group]].to_numpy(), pred[group], W.CAP[group])
        by_group[group] = result
        nmae.append(1.0 - result["one_minus_nmae"])
        ficr.append(result["ficr"])
    total = {
        "score": float(0.5 * (1.0 - np.mean(nmae)) + 0.5 * np.mean(ficr)),
        "one_minus_nmae": float(1.0 - np.mean(nmae)),
        "ficr": float(np.mean(ficr)),
    }
    return by_group, total


def robust_nudges(
    fold_frames: dict[int, dict[int, tuple[pd.DataFrame, pd.DataFrame]]],
    fold_predictions: dict[int, dict[int, np.ndarray]],
) -> dict[int, tuple[float, float]]:
    """Select only group nudges whose score delta is nonnegative in every fold."""
    stores = {}
    for group in V.GROUPS:
        years = [year for year, frames in fold_frames.items() if group in frames]
        if len(years) < 2:
            stores[group] = (1.0, 0.0)
            continue

        cap = W.CAP[group]
        candidates = []
        for scale in np.linspace(0.98, 1.05, 15):
            for shift in np.linspace(-0.01, 0.02, 13) * cap:
                deltas = []
                scores = []
                for year in years:
                    va = fold_frames[year][group][1]
                    actual = va[V.TGT[group]].to_numpy()
                    raw = fold_predictions[year][group]
                    post = np.clip(raw * scale + shift, 0, cap)
                    raw_score = group_result(actual, raw, cap)["score"]
                    post_score = group_result(actual, post, cap)["score"]
                    deltas.append(post_score - raw_score)
                    scores.append(post_score)
                candidates.append((min(deltas), float(np.mean(deltas)), float(np.mean(scores)), scale, shift))

        safe = [candidate for candidate in candidates if candidate[0] >= -1e-12]
        best = max(safe, default=(0.0, 0.0, 0.0, 1.0, 0.0))
        stores[group] = (float(best[3]), float(best[4]))
    return stores


def apply_nudges(
    pred: dict[int, np.ndarray],
    stores: dict[int, tuple[float, float]],
    strengths: dict[int, float] | None = None,
) -> dict[int, np.ndarray]:
    strengths = strengths or {group: 1.0 for group in pred}
    out = {}
    for group, values in pred.items():
        scale, shift = stores[group]
        nudged = np.clip(values * scale + shift, 0, W.CAP[group])
        out[group] = np.clip(
            values + strengths[group] * (nudged - values),
            0,
            W.CAP[group],
        )
    return out


def guarded_nudge_strengths(
    raw: dict[int, np.ndarray],
    stores: dict[int, tuple[float, float]],
) -> dict[int, float]:
    """Retain as much CV-selected nudge as the test mean-CF guard permits."""
    strengths = {}
    for group, values in raw.items():
        cap = W.CAP[group]
        full = apply_nudges({group: values}, {group: stores[group]})[group]
        if 100.0 * np.mean(full) / cap <= FINAL_CF_GUARD:
            strengths[group] = 1.0
            continue
        if 100.0 * np.mean(values) / cap >= FINAL_CF_GUARD:
            strengths[group] = 0.0
            continue

        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            guarded = values + mid * (full - values)
            if 100.0 * np.mean(guarded) / cap <= FINAL_CF_GUARD:
                lo = mid
            else:
                hi = mid
        strengths[group] = lo
    return strengths


def full_frames() -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    frames = {}
    for group in V.GROUPS:
        iso = W.fit_powercurve(V.FR[group], V.TGT[group], W.CAP[group])
        frames[group] = (
            W.with_pc(V.FR[group], iso),
            W.with_pc(V.FR_TE[group], iso),
        )
    return frames


def main() -> None:
    fold_frames = {2023: V.make_2023_frames(), 2024: V.make_2024_frames()}
    fold_predictions = {}
    raw_results = {}
    for year, frames in fold_frames.items():
        print(f"fitting fold {year}")
        fold_predictions[year] = predict_metric_blend(frames)
        by_group, total = evaluate_fold(frames, fold_predictions[year])
        raw_results[year] = {"groups": by_group, "total": total}
        print(year, total)

    stores = robust_nudges(fold_frames, fold_predictions)
    post_results = {}
    for year, frames in fold_frames.items():
        post = apply_nudges(fold_predictions[year], stores)
        by_group, total = evaluate_fold(frames, post)
        post_results[year] = {"groups": by_group, "total": total}
        print(f"{year} post", total)
    print("nudges", stores)

    print("fitting full train")
    raw_test_predictions = predict_metric_blend(full_frames())
    strengths = guarded_nudge_strengths(raw_test_predictions, stores)
    test_predictions = apply_nudges(raw_test_predictions, stores, strengths)
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
    raw_test_cf = {
        group: float(100.0 * np.mean(raw_test_predictions[group]) / W.CAP[group])
        for group in V.GROUPS
    }
    assert max(test_cf.values()) <= FINAL_CF_GUARD + 1e-9
    result = {
        "recipe": {
            "seeds": SEEDS,
            "mlp_weight": MLP_WEIGHT,
            "invalid_gbm_weight": INVALID_WEIGHT,
            "group3_nudge_disabled": True,
            "final_cf_guard": FINAL_CF_GUARD,
        },
        "raw_cv": raw_results,
        "post_cv": post_results,
        "nudges": {str(group): list(value) for group, value in stores.items()},
        "test_nudge_strengths": strengths,
        "raw_test_mean_cf": raw_test_cf,
        "test_mean_cf": test_cf,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("test mean CF", test_cf)
    print(f"saved {SUBMISSION_PATH} and {RESULT_PATH}")


if __name__ == "__main__":
    main()
