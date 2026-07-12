"""v10 conservative forecast-combination experiment.

This keeps the calibrated v7 recipe as the anchor:
- spatial v2 + physics + empirical power curve features
- SCADA stuck-time down-weighting
- tuned LightGBM + HistGBM + pooled MLP
- conservative FICR nudge and final mean-CF guard

The new part is a small OOF-selected forecast combination over candidate
predictions instead of the fixed v7 0.5/0.5 GBM and 0.7 MLP weights.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import wind_lib as W
import wind_pipeline as P
from official_metric import group_scores
from submission_validation import write_submission


DEV = P.DEV
GROUPS = P.GROUPS
STUCK_W = P.STUCK_W
GBM_PARAMS = P.GBM_PARAMS
FR = P.FR
TGT = P.TGT
FR_TE = P.FR_TE
FEATS = P.FEATS
histgbm = P.histgbm
train_one = P.train_one
predict_one = P.predict_one
make_2023_frames = P.make_2023_frames
make_2024_frames = P.make_2024_frames
VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "result.json"
SUBMISSION_PATH = VERSION_DIR / "submission.csv"

FINAL_CF_GUARD = 39.5

OOF_SEEDS = [15, 0, 1]
CV_SEEDS = [15, 0, 1]
FINAL_SEEDS = [15, 0, 1, 2, 3]

@dataclass(frozen=True)
class Combo:
    lgb_weight: float
    mlp_weight: float
    pc_weight: float

def candidate_predict(tr_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]], seeds: list[int]):
    pool = []
    for group, (tr, _) in tr_frames.items():
        p = tr[FEATS + ["kst_dtm", "stuck_mask"]].copy()
        p["cf"] = tr[TGT[group]] / W.CAP[group]
        p["gid"] = group - 1
        p["w"] = np.where(p["stuck_mask"], STUCK_W, 1.0)
        pool.append(p)
    pool_tr = pd.concat(pool, ignore_index=True)

    mlp_acc = {g: [] for g in tr_frames}
    for seed in seeds:
        net, scaler = train_one(pool_tr, seed)
        for group, (_, va) in tr_frames.items():
            mlp_acc[group].append(predict_one(net, scaler, va, group))

    out = {}
    for group, (tr, va) in tr_frames.items():
        cap = W.CAP[group]
        tgt = TGT[group]
        wt = np.where(tr["stuck_mask"], STUCK_W, 1.0)
        lg = lgb.LGBMRegressor(**GBM_PARAMS).fit(tr[FEATS], tr[tgt], sample_weight=wt)
        hg = histgbm().fit(tr[FEATS].to_numpy(), tr[tgt].to_numpy(), sample_weight=wt)
        out[group] = {
            "lgb": np.clip(lg.predict(va[FEATS]), 0, cap),
            "hgb": np.clip(hg.predict(va[FEATS].to_numpy()), 0, cap),
            "mlp": np.clip(np.mean(mlp_acc[group], axis=0), 0, cap),
            "pc": np.clip(va["pc_pred_cf"].to_numpy() * cap, 0, cap),
        }
    return out


def combine_candidates(cands: dict[int, dict[str, np.ndarray]], combo: Combo):
    out = {}
    for group, cand in cands.items():
        cap = W.CAP[group]
        gbm = combo.lgb_weight * cand["lgb"] + (1.0 - combo.lgb_weight) * cand["hgb"]
        raw = (1.0 - combo.mlp_weight) * gbm + combo.mlp_weight * cand["mlp"]
        pred = (1.0 - combo.pc_weight) * raw + combo.pc_weight * cand["pc"]
        out[group] = np.clip(pred, 0, cap)
    return out


def score_predictions(ent, pred):
    nmae, ficr = [], []
    for group, (_, va) in ent.items():
        a, b = group_scores(va[TGT[group]].to_numpy(), pred[group], W.CAP[group])
        nmae.append(a)
        ficr.append(b)
    return 0.5 * (1 - float(np.mean(nmae))) + 0.5 * float(np.mean(ficr)), float(1 - np.mean(nmae)), float(np.mean(ficr))

def oof_candidates_for_group(group: int, tr: pd.DataFrame):
    out = {k: np.full(len(tr), np.nan) for k in ("lgb", "hgb", "mlp", "pc")}
    years = sorted(tr.kst_dtm.dt.year.unique())
    if len(years) >= 2:
        splits = []
        for year in years:
            train_mask = tr.kst_dtm.dt.year != year
            valid_mask = (tr.kst_dtm.dt.year == year).to_numpy()
            splits.append((train_mask.to_numpy(), valid_mask))
    else:
        cut = int(len(tr) * 0.7)
        train_mask = np.zeros(len(tr), dtype=bool)
        valid_mask = np.zeros(len(tr), dtype=bool)
        train_mask[:cut] = True
        valid_mask[cut:] = True
        splits = [(train_mask, valid_mask)]

    for train_mask, valid_mask in splits:
        cand = candidate_predict({group: (tr.iloc[train_mask], tr.iloc[valid_mask])}, OOF_SEEDS)[group]
        for name in out:
            out[name][valid_mask] = cand[name]
    return out


def fit_nudge(yt: np.ndarray, yp: np.ndarray, cap: int, smax: float = 1.05, shmax: float = 0.02):
    best = (1.0, 0.0)
    best_ficr = -1.0
    for scale in np.linspace(2 - smax, smax, 21):
        for shift in np.linspace(-shmax, shmax, 21) * cap:
            _, ficr = group_scores(yt, np.clip(yp * scale + shift, 0, cap), cap)
            if ficr > best_ficr:
                best_ficr = ficr
                best = (float(scale), float(shift))
    return best


def fit_nudges(tr_frames, oof_cands, combo: Combo):
    stores = {}
    for group, (tr, _) in tr_frames.items():
        pred = combine_candidates({group: oof_cands[group]}, combo)[group]
        keep = np.isfinite(pred)
        stores[group] = fit_nudge(tr[TGT[group]].to_numpy()[keep], pred[keep], W.CAP[group])
    return stores


def apply_nudges(pred, stores):
    out = {}
    for group, p in pred.items():
        scale, shift = stores[group]
        out[group] = np.clip(p * scale + shift, 0, W.CAP[group])
    return out


def mean_cf(pred):
    return {str(group): round(float(np.mean(values) / W.CAP[group] * 100), 3) for group, values in pred.items()}


def main():
    print(f"device={DEV} cv_seeds={CV_SEEDS} final_seeds={FINAL_SEEDS}")

    ent23 = make_2023_frames()
    cand23 = candidate_predict(ent23, CV_SEEDS)

    ent24 = make_2024_frames()
    cand24 = candidate_predict(ent24, FINAL_SEEDS)

    print("building OOF candidates for 2024 nudge")
    oof24 = {}
    for group, (tr, _) in ent24.items():
        oof24[group] = oof_candidates_for_group(group, tr)
        finite = int(np.isfinite(oof24[group]["mlp"]).sum())
        print(f"g{group} OOF {finite}/{len(tr)}")

    grid = [
        Combo(lgb_weight=lgb_w, mlp_weight=mlp_w, pc_weight=pc_w)
        for lgb_w in (0.35, 0.50, 0.65)
        for mlp_w in (0.60, 0.65, 0.70, 0.75)
        for pc_w in (0.00, 0.03, 0.06)
    ]

    rows = []
    for combo in grid:
        pred23 = combine_candidates(cand23, combo)
        s23, one23, f23 = score_predictions(ent23, pred23)

        raw24 = combine_candidates(cand24, combo)
        stores = fit_nudges(ent24, oof24, combo)
        post24 = apply_nudges(raw24, stores)
        s24, one24, f24 = score_predictions(ent24, post24)
        rows.append(
            {
                **asdict(combo),
                "score23_raw": round(s23, 6),
                "one23_raw": round(one23, 6),
                "ficr23_raw": round(f23, 6),
                "score24_post": round(s24, 6),
                "one24_post": round(one24, 6),
                "ficr24_post": round(f24, 6),
                "holdout_cf": mean_cf(post24),
                "stores": {str(g): [round(stores[g][0], 6), round(stores[g][1], 3)] for g in stores},
            }
        )

    current = next(r for r in rows if r["lgb_weight"] == 0.5 and r["mlp_weight"] == 0.7 and r["pc_weight"] == 0.0)

    def rank_key(row):
        guard_ok = max(row["holdout_cf"].values()) <= FINAL_CF_GUARD
        robust_delta = min(row["score23_raw"] - current["score23_raw"], row["score24_post"] - current["score24_post"])
        return (guard_ok, robust_delta, row["score24_post"], -row["pc_weight"], -abs(row["mlp_weight"] - 0.7))

    best = max(rows, key=rank_key)
    if rank_key(best)[1] < -0.0005:
        best = current

    best_combo = Combo(
        lgb_weight=float(best["lgb_weight"]),
        mlp_weight=float(best["mlp_weight"]),
        pc_weight=float(best["pc_weight"]),
    )
    best_stores = {int(g): tuple(v) for g, v in best["stores"].items()}

    print("current", json.dumps(current, ensure_ascii=False))
    print("best", json.dumps(best, ensure_ascii=False))

    full_frames = {}
    for group in GROUPS:
        iso = W.fit_powercurve(FR[group], TGT[group], W.CAP[group])
        full_frames[group] = (W.with_pc(FR[group], iso), W.with_pc(FR_TE[group], iso))
    cand_test = candidate_predict(full_frames, FINAL_SEEDS)
    pred_test = apply_nudges(combine_candidates(cand_test, best_combo), best_stores)

    out = W.load_test(1)[["forecast_id", "kst_dtm"]].rename(columns={"kst_dtm": "forecast_kst_dtm"})
    for group in GROUPS:
        out[f"kpx_group_{group}"] = pred_test[group]
    assert out.shape[0] == 8760
    for group in GROUPS:
        c = out[f"kpx_group_{group}"]
        assert c.notna().all() and (c >= 0).all() and (c <= W.CAP[group]).all()

    bias = {str(group): round(float(out[f"kpx_group_{group}"].mean() / W.CAP[group] * 100), 3) for group in GROUPS}
    result = {
        "current": current,
        "best": best,
        "selection_note": "Chosen by robust delta vs fixed v7-style combo, with mean-CF guard.",
        "final_mean_cf_pct": bias,
        "grid": sorted(rows, key=lambda r: (r["score24_post"], r["score23_raw"]), reverse=True),
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_submission(out, SUBMISSION_PATH)

    print(f"saved {SUBMISSION_PATH}", out.shape)
    print("final mean CF (%)", bias)
    assert max(bias.values()) <= FINAL_CF_GUARD, f"mean-CF guard failed: {bias}"


if __name__ == "__main__":
    main()
