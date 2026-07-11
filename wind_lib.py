"""BARAM 2026 풍력 예측 공유 라이브러리.

누설 방지 원칙(문서 CONSTRAINTS §0):
- 테스트 예측 = 테스트 NWP 입력 + 학습구간에서 배운 파라미터의 함수.
- 파워커브/HMM/모델은 모두 학습구간에서만 fit → 검증/테스트 NWP에 적용.
- 실측 발전량·SCADA는 타깃/파워커브/라벨정제 외 용도로 절대 미사용.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

CAP = {1: 21600, 2: 21600, 3: 21000}          # 시간당 설비용량 kWh
VALID_START = pd.Timestamp("2024-01-01")
VALID_END = pd.Timestamp("2025-01-01")         # 학습라벨 끝 (2025 경계행 배제)
VALID_CF = 0.10                                # 대회 채점: 실측 CF >= 10%
EPS = 1e-6

_CAND = [Path("preprocessed"), Path("/root/wind/preprocessed"),
         Path(__file__).resolve().parent / "preprocessed"]


def data_dir() -> Path:
    for p in _CAND:
        if p.exists():
            return p
    raise FileNotFoundError(f"preprocessed not found; tried {_CAND}")


def load_train(g, data=None):
    data = data or data_dir()
    df = pd.read_parquet(data / f"train_kpx_group_{g}.parquet")
    tgt = f"kpx_group_{g}"
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
    df = df.dropna(subset=[tgt]).copy()
    df = df[df.kst_dtm < VALID_END]
    return df.sort_values("kst_dtm").reset_index(drop=True), tgt


def load_test(g, data=None):
    data = data or data_dir()
    return pd.read_parquet(data / f"test_kpx_group_{g}.parquet").sort_values("kst_dtm").reset_index(drop=True)


def strip(df, g):
    """그룹 prefix 제거 → 그룹 간 컬럼명 통일."""
    p = f"kpx_group_{g}_"
    return df.rename(columns={c: c[len(p):] for c in df.columns if c.startswith(p)})


def add_physics(d):
    """NWP 컬럼만으로 물리 파생 추가 (누설 없음)."""
    d = d.copy()
    v100 = d["gfs_wind_speed_100m_mean"]; v80 = d["gfs_wind_speed_80m_mean"]
    v10 = d["gfs_wind_speed_10m_mean"]
    l50 = d["ldaps_wind_speed_50m_max_mean"]; l10 = d["ldaps_wind_speed_10m_mean"]
    gust = d["gfs_surface_0_gust_mean"]
    T = d["gfs_heightAboveGround_2_2t_mean"]; P = d["gfs_surface_0_sp_mean"]
    d["hub_v"] = v100; d["hub_v2"] = v100**2; d["hub_v3"] = v100**3
    d["shear_gfs"] = v100 / (v10 + EPS)
    d["alpha_gfs"] = np.log((v100 + EPS) / (v10 + EPS)) / np.log(100 / 10)
    d["shear_ldaps"] = l50 / (l10 + EPS)
    rho = P / (287.05 * T)
    d["air_density"] = rho; d["rho_v3"] = rho * v100**3
    d["gust_ratio"] = gust / (v100 + EPS); d["gust_excess"] = gust - v100
    d["gfs_ldaps_mean"] = 0.5 * (v80 + l50); d["gfs_ldaps_diff"] = v80 - l50
    u = d["gfs_heightAboveGround_100_100u_mean"]; v = d["gfs_heightAboveGround_100_100v_mean"]
    wd = np.arctan2(-u, -v)
    d["wdir_sin"] = np.sin(wd); d["wdir_cos"] = np.cos(wd)
    h = d["hour"].to_numpy()
    d["lead_h"] = np.where(h == 0, 39, h + 15)
    return d


PHYS_COLS = ["hub_v", "hub_v2", "hub_v3", "shear_gfs", "alpha_gfs", "shear_ldaps",
             "air_density", "rho_v3", "gust_ratio", "gust_excess", "gfs_ldaps_mean",
             "gfs_ldaps_diff", "wdir_sin", "wdir_cos", "lead_h"]


def build(df, g):
    return add_physics(strip(df, g))


def feature_cols(fr):
    non = {"kst_dtm", "forecast_id"} | {f"kpx_group_{g}" for g in (1, 2, 3)}
    return [c for c in fr.columns if c not in non]


def fit_powercurve(fr, tgt, cap, wind="hub_v"):
    d = pd.DataFrame({"ws": fr[wind].to_numpy(), "cf": (fr[tgt] / cap).to_numpy()}).dropna()
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True)
    iso.fit(d.ws, d.cf)
    return iso


def with_pc(fr, iso):
    fr = fr.copy(); fr["pc_pred_cf"] = iso.predict(fr["hub_v"].to_numpy()); return fr


def nmae_valid(y_true, y_pred, cap):
    """대회 지표: 유효구간(실측 CF>=10%)에서 MAE/설비용량 (%)."""
    y_true = np.asarray(y_true, float); y_pred = np.clip(np.asarray(y_pred, float), 0, cap)
    m = y_true >= VALID_CF * cap
    return (mean_absolute_error(y_true[m], y_pred[m]) / cap * 100, int(m.sum())) if m.sum() else (np.nan, 0)


# ── v2: 공간·안정도 feature (scripts/build_spatial_v2.py 산출) ──────────────
# 근거: claudedocs/research_nwp_features_2026-07-09.md (다중격자 ★★★, 감률 ★★☆)
SPATIAL_COLS = ["gfs_ws100_grid_mean", "gfs_ws100_grid_std", "gfs_ws100_grad_ew",
                "gfs_ws100_grad_ns", "gfs_lapse_850_700", "gfs_inversion_2m_850",
                "ldaps_ws10_grid_mean", "ldaps_ws10_grid_std", "ldaps_ws10_grad_ew",
                "ldaps_ws10_grad_ns"]

# FEATURE_REDUCTION.ipynb F3 축소 세트의 제거 목록 (CV 검증 완료)
TREE_INVARIANT = ["hub_v", "hub_v2", "hub_v3"]
DEAD_NWP = [
    "ldaps_heightAboveGround_2_dpt_mean", "ldaps_heightAboveGround_2_r_mean",
    "ldaps_heightAboveGround_2_q_mean", "ldaps_etc_0_hcc_mean", "ldaps_etc_0_mcc_mean",
    "ldaps_etc_0_lcc_mean", "ldaps_etc_0_VLCDC_mean", "ldaps_surface_0_avg_lsprate_mean",
    "ldaps_surface_0_lssrate_mean", "ldaps_surface_0_ncpcp_mean",
    "gfs_heightAboveGround_2_2d_mean", "gfs_heightAboveGround_2_2r_mean",
    "gfs_heightAboveGround_2_2sh_mean", "gfs_surface_0_prate_mean", "gfs_surface_0_tp_mean",
    "gfs_lowCloudLayer_0_lcc_mean", "gfs_middleCloudLayer_0_mcc_mean",
    "gfs_highCloudLayer_0_hcc_mean", "gfs_atmosphere_0_tcc_mean"]
WIND_REDUNDANT = [
    "ldaps_wind_speed_10m_std", "ldaps_wind_speed_10m_min", "ldaps_wind_speed_10m_max",
    "ldaps_wind_speed_50m_max_std", "ldaps_wind_speed_50m_max_min", "ldaps_wind_speed_50m_max_max",
    "ldaps_wind_speed_50m_min_std", "ldaps_wind_speed_50m_min_min", "ldaps_wind_speed_50m_min_max"]


def load_spatial(split, data=None):
    """v2 공간·안정도 feature (kst_dtm 키, 3그룹 공통)."""
    data = data or data_dir()
    return pd.read_parquet(data / f"spatial_v2_{split}.parquet")


def add_spatial(fr, split, data=None):
    """build() 결과에 v2 feature 조인."""
    sp = load_spatial(split, data)
    out = fr.merge(sp, on="kst_dtm", how="left")
    assert out[SPATIAL_COLS].notna().all().all(), "spatial join NaN"
    return out


def lean_features(cols, include_pc=True):
    """F3 축소 세트 (86→55) + optional pc_pred_cf."""
    rm = set(TREE_INVARIANT) | set(DEAD_NWP) | set(WIND_REDUNDANT)
    out = [c for c in cols if c not in rm]
    if include_pc and "pc_pred_cf" not in out:
        out.append("pc_pred_cf")
    return out


# ── v9: 시간 이웃 NWP feature (같은 예보 배치 내 lag/lead — 누설·타이밍 안전) ──
# 배치 = 전일 13시 공개된 01:00~24:00 24시간. 배치 경계를 넘는 lead는 미래 배치
# (더 늦게 공개) 참조라 금지 → 경계에서는 자기값으로 채움.
TEMPORAL_BASE = ["gfs_wind_speed_100m_mean", "ldaps_wind_speed_10m_mean"]
TEMPORAL_COLS = [
    "tmp_gfs_lag1", "tmp_gfs_lead1", "tmp_gfs_lag3", "tmp_gfs_lead3",
    "tmp_gfs_bmean", "tmp_gfs_bstd", "tmp_gfs_anom", "tmp_gfs_trend",
    "tmp_ldaps_lag1", "tmp_ldaps_lead1", "tmp_ldaps_bmean", "tmp_ldaps_anom"]


def add_temporal(fr):
    """같은 예보 배치 내 시간 이웃 feature. NWP 타이밍 오차(전선 지연 등) 평활·추세."""
    fr = fr.sort_values("kst_dtm").reset_index(drop=True).copy()
    batch = (fr["kst_dtm"] - pd.Timedelta(hours=1)).dt.floor("D")
    for c, pre in [(TEMPORAL_BASE[0], "gfs"), (TEMPORAL_BASE[1], "ldaps")]:
        g = fr.groupby(batch)[c]
        fr[f"tmp_{pre}_lag1"] = g.shift(1).fillna(fr[c])
        fr[f"tmp_{pre}_lead1"] = g.shift(-1).fillna(fr[c])
        if pre == "gfs":
            fr[f"tmp_{pre}_lag3"] = g.shift(3).fillna(fr[c])
            fr[f"tmp_{pre}_lead3"] = g.shift(-3).fillna(fr[c])
        bm = g.transform("mean")
        fr[f"tmp_{pre}_bmean"] = bm
        if pre == "gfs":
            fr[f"tmp_{pre}_bstd"] = g.transform("std").fillna(0.0)
        fr[f"tmp_{pre}_anom"] = fr[c] - bm
        if pre == "gfs":
            fr[f"tmp_{pre}_trend"] = fr[f"tmp_{pre}_lead1"] - fr[f"tmp_{pre}_lag1"]
    assert fr[TEMPORAL_COLS].notna().all().all()
    return fr


# ── NWP-only HMM 국면(regime) ─────────────────────────────────────────────
# 규칙(research 문서 §0): regime은 반드시 NWP 파생 변수로만 정의. 실측 발전량 사용 금지.
REGIME_VARS = ["hub_v", "shear_gfs", "alpha_gfs", "gust_ratio", "air_density",
               "gfs_ldaps_diff", "gfs_wind_speed_850hpa_mean"]


def fit_regime_hmm(fr_train, n_states=4, seed=42, cols=None):
    """학습구간 NWP만으로 Gaussian HMM 적합. (scaler, hmm, cols) 반환."""
    from hmmlearn.hmm import GaussianHMM
    cols = cols or REGIME_VARS
    X = fr_train[cols].to_numpy()
    mu = X.mean(0); sd = X.std(0) + EPS
    Xs = (X - mu) / sd
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                      n_iter=200, random_state=seed, tol=1e-3)
    hmm.fit(Xs)
    return (mu, sd, cols), hmm


def regime_posteriors(fr, scaler, hmm):
    """soft posterior(감마) K열 반환. 하드 argmax 금지(research §3)."""
    mu, sd, cols = scaler
    Xs = (fr[cols].to_numpy() - mu) / sd
    post = hmm.predict_proba(Xs)
    return pd.DataFrame(post, columns=[f"regime_{k}" for k in range(post.shape[1])], index=fr.index)
