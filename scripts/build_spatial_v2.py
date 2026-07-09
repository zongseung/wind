"""전처리 v2 보강 feature 생성 — 원본(open/)에서 공간·안정도 feature.

리서치 근거 (claudedocs/research_nwp_features_2026-07-09.md):
- 다중 격자 공간 feature: Andrade & Bessa 2017, 풍력 MAE -12.85% (★★★)
- 기온 감률(안정도): 물리·자원평가 근거 (★★☆) — t850-t700, t2m-t850 2개만

산출: preprocessed/spatial_v2_train.parquet, spatial_v2_test.parquet
      (kst_dtm 키, 3그룹 공통 — 격자 전역 통계이므로)

누설 안전: 원본은 전일 13:00 공개 예보만 포함(검증 완료, PREPROCESSING_VERIFICATION.md).
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("/Users/ijongseung/Downloads/open")
OUT = Path(__file__).resolve().parent.parent / "preprocessed"


def gfs_features(path):
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    df["ws100"] = np.hypot(df["heightAboveGround_100_100u"], df["heightAboveGround_100_100v"])
    piv = df.pivot_table(index="forecast_kst_dtm", columns="grid_id", values="ws100")
    out = pd.DataFrame(index=piv.index)
    # 9격자 공간 통계
    out["gfs_ws100_grid_mean"] = piv.mean(axis=1)
    out["gfs_ws100_grid_std"] = piv.std(axis=1, ddof=1)
    # 중앙 통과 gradient (grid 배치: 1,2,3 / 4,5,6 / 7,8,9; 위→아래 = 북→남)
    out["gfs_ws100_grad_ew"] = (piv[6] - piv[4]) / 0.5      # 동-서 (per degree lon)
    out["gfs_ws100_grad_ns"] = (piv[2] - piv[8]) / 0.5      # 북-남 (per degree lat)
    # 감률(안정도): 최근접 격자(5) 기준 — 기존 전처리와 동일한 지점
    g5 = df[df.grid_id == 5].set_index("forecast_kst_dtm")
    out["gfs_lapse_850_700"] = g5["isobaricInhPa_850_t"] - g5["isobaricInhPa_700_t"]
    out["gfs_inversion_2m_850"] = g5["heightAboveGround_2_2t"] - g5["isobaricInhPa_850_t"]
    return out


def ldaps_features(path):
    usecols = ["forecast_kst_dtm", "grid_id", "latitude", "longitude",
               "heightAboveGround_10_10u", "heightAboveGround_10_10v"]
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"], usecols=usecols)
    df["ws10"] = np.hypot(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
    piv = df.pivot_table(index="forecast_kst_dtm", columns="grid_id", values="ws10")
    gg = df[["grid_id", "latitude", "longitude"]].drop_duplicates().set_index("grid_id")
    out = pd.DataFrame(index=piv.index)
    out["ldaps_ws10_grid_mean"] = piv.mean(axis=1)
    out["ldaps_ws10_grid_std"] = piv.std(axis=1, ddof=1)
    # gradient: 최동단열 평균 - 최서단열 평균 / 경도차 (16격자 불규칙 대비 일반형)
    east = gg.longitude >= gg.longitude.median()
    north = gg.latitude >= gg.latitude.median()
    dlon = gg.longitude[east].mean() - gg.longitude[~east].mean()
    dlat = gg.latitude[north].mean() - gg.latitude[~north].mean()
    out["ldaps_ws10_grad_ew"] = (piv[gg.index[east]].mean(axis=1) - piv[gg.index[~east]].mean(axis=1)) / dlon
    out["ldaps_ws10_grad_ns"] = (piv[gg.index[north]].mean(axis=1) - piv[gg.index[~north]].mean(axis=1)) / dlat
    return out


def build(split):
    g = gfs_features(RAW / split / f"gfs_{split}.csv")
    l = ldaps_features(RAW / split / f"ldaps_{split}.csv")
    out = g.join(l, how="inner").reset_index().rename(columns={"forecast_kst_dtm": "kst_dtm"})
    assert out.notna().all().all(), f"{split}: NaN 존재"
    path = OUT / f"spatial_v2_{split}.parquet"
    out.to_parquet(path, index=False)
    print(f"{split}: {out.shape} -> {path}")
    print(out.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    return out


if __name__ == "__main__":
    build("train")
    build("test")
