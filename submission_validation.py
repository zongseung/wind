"""Validation and atomic writing for competition submission files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import wind_lib as W


KEY_COLUMNS = ["forecast_id", "forecast_kst_dtm"]
TARGET_COLUMNS = [f"kpx_group_{group}" for group in W.CAP]
SUBMISSION_COLUMNS = KEY_COLUMNS + TARGET_COLUMNS
EXPECTED_ROWS = 8_760


def validate_submission(frame: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> None:
    """Raise ``ValueError`` when a submission violates the competition contract."""
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"submission columns must be {SUBMISSION_COLUMNS}, got {list(frame.columns)}"
        )
    if len(frame) != expected_rows:
        raise ValueError(f"submission must have {expected_rows} rows, got {len(frame)}")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError("submission keys contain missing values")
    if frame["forecast_id"].duplicated().any():
        raise ValueError("forecast_id must be unique")
    for group, capacity in W.CAP.items():
        column = f"kpx_group_{group}"
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
        if ((values < 0) | (values > capacity)).any():
            raise ValueError(f"{column} contains values outside [0, {capacity}]")


def write_submission(frame: pd.DataFrame, path: Path) -> None:
    """Validate and atomically replace a submission CSV."""
    validate_submission(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
