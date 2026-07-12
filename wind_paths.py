"""Portable project paths configured through environment variables."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else default


def preprocessed_dir() -> Path:
    """Return the prepared parquet directory.

    Override with ``WIND_PREPROCESSED_DIR`` when the data is stored elsewhere.
    """
    return _configured_path("WIND_PREPROCESSED_DIR", PROJECT_ROOT / "preprocessed")


def raw_data_dir() -> Path:
    """Return the competition raw-data directory.

    Override with ``WIND_RAW_DIR``.  The default matches the documented local
    layout without embedding a user name in source code.
    """
    return _configured_path("WIND_RAW_DIR", Path.home() / "Downloads" / "open")
