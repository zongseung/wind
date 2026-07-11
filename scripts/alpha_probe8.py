"""FICR-정렬 가중 α 확장 스캔 {3, 5} — α=2가 스캔 경계 채택이라 단조성 확인.

출력: final_alpha.json (v8 파이프라인이 읽음)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
torch.set_num_threads(1)
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
import wind_lib as W
from official_metric import group_scores

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = [15, 0, 1]; BLEND = 0.7; STUCK_TH = 0.3; STUCK_W = 0.5
MLP_CFG = dict(h=256, depth=3, drop=0.0, lr=0.0015868563457489381, wd=1e-4, bs=256, emb=4)
GBM_PARAMS = dict(objective="mae", n_estimators=2000, learning_rate=0.020651822836313095,
    num_leaves=63, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    reg_lambda=0.1, random_state=42, n_jobs=1, verbose=-1)
RAW = "/Users/ijongseung/Downloads/open"
GROUPS = (1, 2, 3)

FR, TGT = {}, {}
for g in GROUPS:
    df, tgt = W.load_train(g); TGT[g] = tgt
    FR[g] = W.add_spatial(W.build(df, g), "train")
BASE_ALL = [c for c in W.feature_cols(FR[1]) if c not in W.SPATIAL_COLS] + ["pc_pred_cf"]
FEATS = W.lean_features(BASE_ALL) + W.SPATIAL_COLS


def stuck_frac():
    frames = {}
    for fn, pre, n, rate in [("scada_vestas_train.csv", "vestas", 12, 3600.0),
                             ("scada_unison_train.csv", "unison", 5, 4200.0)]:
        d = pd.read_csv(f"{RAW}/train/{fn}", encoding="utf-8-sig", parse_dates=["kst_dtm"])
        d["hour"] = d["kst_dtm"].dt.ceil("h")
        agg = {}
        for i in range(1, n + 1):
            pw = f"{pre}_wtg{i:02d}_power_kw10m"; ws = f"{pre}_wtg{i:02d}_ws"
            h = d.groupby("hour").agg(pw_m=(pw, "mean"), ws_m=(ws, "mean"))
            agg[i] = pd.DataFrame({f"stuck_{i}": ((h.ws_m >= 4.0) & (h.pw_m <= 0.01 * rate)).astype(float),
                                   f"rep_{i}": h.pw_m.notna().astype(float)})
        frames[pre] = pd.concat(agg.values(), axis=1)
    def frac(pre, ids):
        f = frames[pre]
        st = f[[f"stuck_{i}" for i in ids]].sum(axis=1); rp = f[[f"rep_{i}" for i in ids]].sum(axis=1)
        return (st / rp).where(rp >= 3)
    return {1: frac("vestas", range(1, 7)), 2: frac("vestas", range(7, 13)), 3: frac("unison", range(1, 6))}


FRAC = stuck_frac()
for g in GROUPS:
    s = FRAC[g].reindex(FR[g].kst_dtm).to_numpy()
    FR[g] = FR[g].assign(stuck_frac=np.nan_to_num(s, nan=0.0))


def make_weight(fr, tgt, cap, alpha):
    w = np.where(fr["stuck_frac"] >= STUCK_TH, STUCK_W, 1.0)
    if alpha > 0:
        w = w * (1.0 + alpha * np.clip(fr[tgt].to_numpy() / cap, 0, 1))
    return w


class MLP(nn.Module):
    def __init__(s, nf, ng=3, h=256, depth=3, drop=0.0, emb=4):
        super().__init__(); s.emb = nn.Embedding(ng, emb)
        L = [nn.Linear(nf + emb, h), nn.GELU(), nn.Dropout(drop)]
        for _ in range(depth - 1): L += [nn.Linear(h, h), nn.GELU(), nn.Dropout(drop)]
        L += [nn.Linear(h, 1)]; s.net = nn.Sequential(*L)
    def forward(s, x, g): return s.net(torch.cat([x, s.emb(g)], -1)).squeeze(-1)


def train_one(pool_tr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    pool_tr = pool_tr.sort_values("kst_dtm")
    mu = pool_tr[FEATS].mean(); sd = pool_tr[FEATS].std() + 1e-8
    X = ((pool_tr[FEATS] - mu) / sd).to_numpy(np.float32)
    y = pool_tr["cf"].to_numpy(np.float32); gid = pool_tr["gid"].to_numpy(np.int64)
    wt = pool_tr["w"].to_numpy(np.float32)
    n = len(X); cut = int(n * 0.85)
    Xt = torch.tensor(X, device=DEV); yt = torch.tensor(y, device=DEV)
    gt = torch.tensor(gid, device=DEV); wtt = torch.tensor(wt, device=DEV)
    net = MLP(len(FEATS), **{k: MLP_CFG[k] for k in ("h", "depth", "drop", "emb")}).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_CFG["lr"], weight_decay=MLP_CFG["wd"])
    best = 1e9; st = None; pat = 0
    for ep in range(120):
        net.train(); perm = np.random.permutation(np.arange(cut))
        for i in range(0, len(perm), MLP_CFG["bs"]):
            b = torch.tensor(perm[i:i + MLP_CFG["bs"]], device=DEV); opt.zero_grad()
            e = (net(Xt[b], gt[b]) - yt[b]).abs()
            ((e * wtt[b]).sum() / (wtt[b].sum() + 1e-8)).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            e = (net(Xt[cut:], gt[cut:]) - yt[cut:]).abs()
            vl = ((e * wtt[cut:]).sum() / (wtt[cut:].sum() + 1e-8)).item()
        if vl < best - 1e-5: best = vl; st = {k: v.clone() for k, v in net.state_dict().items()}; pat = 0
        else: pat += 1
        if pat >= 10: break
    net.load_state_dict(st); return net, (mu, sd)


def predict_one(net, scaler, fr, g, cap):
    mu, sd = scaler
    X = torch.tensor(((fr[FEATS] - mu) / sd).to_numpy(np.float32), device=DEV)
    gid = torch.full((len(fr),), g - 1, dtype=torch.long, device=DEV)
    net.eval()
    with torch.no_grad(): p = net(X, gid).cpu().numpy()
    return np.clip(p, 0, 1) * cap


FOLDS = {2023: [2022], 2024: [2022, 2023]}
CACHE = {}
for vy, tys in FOLDS.items():
    ent = {}
    for g in GROUPS:
        tgt = TGT[g]; cap = W.CAP[g]; fr = FR[g]; yr = fr.kst_dtm.dt.year
        tr = fr[yr.isin(tys)]; va = fr[yr == vy]
        if len(tr) == 0 or len(va) == 0: continue
        iso = W.fit_powercurve(tr, tgt, cap)
        ent[g] = (W.with_pc(tr, iso), W.with_pc(va, iso))
    CACHE[vy] = ent


def run_alpha(alpha):
    out = {}
    for vy, ent in CACHE.items():
        gbm = {}
        for g, (tr2, va2) in ent.items():
            cap = W.CAP[g]; tgt = TGT[g]; w = make_weight(tr2, tgt, cap, alpha)
            lg_ = lgb.LGBMRegressor(**GBM_PARAMS).fit(tr2[FEATS], tr2[tgt], sample_weight=w)
            hg_ = HGB(loss="absolute_error", max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                      l2_regularization=1.0, random_state=42).fit(tr2[FEATS].to_numpy(), tr2[tgt].to_numpy(), sample_weight=w)
            gbm[g] = np.clip(0.5 * (lg_.predict(va2[FEATS]) + hg_.predict(va2[FEATS].to_numpy())), 0, cap)
        pool = []
        for g, (tr2, _) in ent.items():
            p = tr2[FEATS + ["kst_dtm"]].copy()
            p["cf"] = tr2[TGT[g]] / W.CAP[g]; p["gid"] = g - 1
            p["w"] = make_weight(tr2, TGT[g], W.CAP[g], alpha); pool.append(p)
        pool = pd.concat(pool, ignore_index=True)
        acc = {g: [] for g in ent}
        for sd_ in SEEDS:
            net, scaler = train_one(pool, sd_)
            for g, (_, va2) in ent.items():
                acc[g].append(predict_one(net, scaler, va2, g, W.CAP[g]))
        nm = []; fi = []
        for g, (_, va2) in ent.items():
            cap = W.CAP[g]
            p = np.clip((1 - BLEND) * gbm[g] + BLEND * np.mean(acc[g], axis=0), 0, cap)
            a, b = group_scores(va2[TGT[g]].to_numpy(), p, cap); nm.append(a); fi.append(b)
        out[vy] = 0.5 * (1 - np.mean(nm)) + 0.5 * np.mean(fi)
    return out


REF2 = {2023: 0.6193, 2024: 0.6327}   # α=5 (alpha_ext 실측)
res = {5.0: REF2}
for a in [8.0]:
    res[a] = run_alpha(a)
    print(f"α={a}: 2023={res[a][2023]:.4f}  2024={res[a][2024]:.4f}", flush=True)

best = 5.0
for a in [8.0]:
    if res[a][2023] >= res[best][2023] and res[a][2024] >= res[best][2024]:
        best = a
print(f"최종 α = {best}")
json.dump(dict(final_alpha=best,
               scan={str(a): {str(k): round(v, 4) for k, v in r.items()} for a, r in res.items()}),
          open("final_alpha.json", "w"), ensure_ascii=False, indent=2)
print("saved final_alpha.json")
