"""Rejected within-forecast-batch temporal features retained for replay."""

from __future__ import annotations

import pandas as pd


TEMPORAL_BASE = ["gfs_wind_speed_100m_mean", "ldaps_wind_speed_10m_mean"]
TEMPORAL_COLS = [
    "tmp_gfs_lag1",
    "tmp_gfs_lead1",
    "tmp_gfs_lag3",
    "tmp_gfs_lead3",
    "tmp_gfs_bmean",
    "tmp_gfs_bstd",
    "tmp_gfs_anom",
    "tmp_gfs_trend",
    "tmp_ldaps_lag1",
    "tmp_ldaps_lead1",
    "tmp_ldaps_bmean",
    "tmp_ldaps_anom",
]


def add_temporal(frame):
    """Add lag, lead, batch-summary and trend features within each forecast batch."""
    frame = frame.sort_values("kst_dtm").reset_index(drop=True).copy()
    batch = (frame["kst_dtm"] - pd.Timedelta(hours=1)).dt.floor("D")
    for column, prefix in zip(TEMPORAL_BASE, ("gfs", "ldaps")):
        grouped = frame.groupby(batch)[column]
        frame[f"tmp_{prefix}_lag1"] = grouped.shift(1).fillna(frame[column])
        frame[f"tmp_{prefix}_lead1"] = grouped.shift(-1).fillna(frame[column])
        if prefix == "gfs":
            frame[f"tmp_{prefix}_lag3"] = grouped.shift(3).fillna(frame[column])
            frame[f"tmp_{prefix}_lead3"] = grouped.shift(-3).fillna(frame[column])
        batch_mean = grouped.transform("mean")
        frame[f"tmp_{prefix}_bmean"] = batch_mean
        if prefix == "gfs":
            frame[f"tmp_{prefix}_bstd"] = grouped.transform("std").fillna(0.0)
        frame[f"tmp_{prefix}_anom"] = frame[column] - batch_mean
        if prefix == "gfs":
            frame[f"tmp_{prefix}_trend"] = (
                frame[f"tmp_{prefix}_lead1"] - frame[f"tmp_{prefix}_lag1"]
            )
    if not frame[TEMPORAL_COLS].notna().all().all():
        raise ValueError("temporal feature generation produced missing values")
    return frame
