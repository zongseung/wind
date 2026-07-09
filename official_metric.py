"""BARAM 2026 공식 채점 지표 (대회 제공 코드 그대로).

총점 = 0.5·(1-NMAE) + 0.5·FICR
- NMAE  = 그룹별 (유효시간 절대오차율 평균)의 그룹평균.  유효시간 = 실측 >= 설비용량 10%.
- FICR  = 그룹별 [sum(실측·단가)/sum(실측·4)]의 그룹평균.  발전량 가중 2티어.
          단가: 오차율<=6% → 4.0, <=8% → 3.0, 그 외 0.0.
"""
import numpy as np

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}


def group_scores(actual, forecast, capacity):
    """단일 그룹의 (nmae_fraction, ficr) 반환. 유효구간만."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    valid = actual >= capacity * 0.10
    actual = actual[valid]
    forecast = forecast[valid]
    if actual.size == 0:
        return np.nan, np.nan

    error_rate = np.abs(forecast - actual) / capacity
    nmae = float(np.mean(error_rate))

    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    earned = float(np.sum(actual * unit_price))
    maximum = float(np.sum(actual * 4.0))
    ficr = earned / maximum if maximum > 0 else np.nan
    return nmae, ficr


def metric(answer_df, pred_df):
    """대회 제공 metric 그대로. (total_score, one_minus_nmae, ficr)."""
    group_nmae = []
    group_ficr = []
    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]

        valid = actual >= capacity * 0.10
        actual = actual[valid]
        forecast = forecast[valid]

        error_rate = np.abs(forecast - actual) / capacity
        group_nmae.append(np.mean(error_rate))

        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        earned_settlement = np.sum(actual * unit_price)
        max_settlement = np.sum(actual * 4.0)
        group_ficr.append(earned_settlement / max_settlement)

    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr
    return total_score, one_minus_nmae, ficr
