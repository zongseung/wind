"""Strict backtest for compact spatial features and a pooled group-3 GBM.

The experiment follows the repository constraints before changing the v12
submission recipe:

- inputs are only the provided GFS/LDAPS forecasts and static info.xlsx data;
- raw grids are compressed to site IDW summaries or fold-fitted PCA scores;
- annual expanding folds choose at most one candidate per group;
- quarterly expanding folds and held-out monthly slices gate adoption.

This script is diagnostic.  It writes no submission file.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import wind_lib as W
import wind_pipeline as V
from submission.ver_11 import pipeline as M
from wind_paths import raw_data_dir


RAW = raw_data_dir()
VERSION_DIR = Path(__file__).resolve().parent
RESULT_PATH = VERSION_DIR / "spatial_backtest.json"
SITE_NEIGHBORS = 4
PCA_COMPONENTS = 3
MIN_ANNUAL_DELTA = 3e-4
MIN_POSITIVE_FRACTION = 0.60

# Official turbine coordinates transcribed from info.xlsx.  Each row is
# latitude degrees/minutes/seconds followed by longitude degrees/minutes/seconds.
TURBINE_DMS = {
    1: [
        (37, 16, 55.61, 128, 57, 2.10),
        (37, 17, 4.05, 128, 56, 58.35),
        (37, 17, 11.49, 128, 56, 58.99),
        (37, 17, 23.11, 128, 57, 3.68),
        (37, 17, 28.20, 128, 57, 15.58),
        (37, 17, 19.48, 128, 57, 24.96),
    ],
    2: [
        (37, 17, 16.20, 128, 57, 34.67),
        (37, 17, 11.29, 128, 57, 47.24),
        (37, 17, 0.97, 128, 57, 57.44),
        (37, 16, 52.77, 128, 58, 4.18),
        (37, 16, 44.89, 128, 58, 1.12),
        (37, 16, 30.58, 128, 58, 2.54),
    ],
    3: [
        (37, 16, 59.73, 128, 57, 44.97),
        (37, 16, 40.41, 128, 58, 13.80),
        (37, 16, 28.03, 128, 58, 22.54),
        (37, 16, 18.58, 128, 58, 29.01),
        (37, 16, 6.83, 128, 58, 35.68),
    ],
}

SOURCE_SPECS = {
    "gfs": {
        "file": "gfs_{split}.csv",
        "u": "heightAboveGround_100_100u",
        "v": "heightAboveGround_100_100v",
    },
    "ldaps": {
        "file": "ldaps_{split}.csv",
        "u": "heightAboveGround_10_10u",
        "v": "heightAboveGround_10_10v",
    },
}


def decimal_coordinates(rows: Iterable[tuple[float, ...]]) -> np.ndarray:
    out = []
    for lat_d, lat_m, lat_s, lon_d, lon_m, lon_s in rows:
        out.append(
            (
                lat_d + lat_m / 60.0 + lat_s / 3600.0,
                lon_d + lon_m / 60.0 + lon_s / 3600.0,
            )
        )
    return np.asarray(out, dtype=float)


def site_weights(
    grid_coordinates: np.ndarray,
    turbine_coordinates: np.ndarray,
    neighbors: int = SITE_NEIGHBORS,
) -> np.ndarray:
    """Average inverse-distance weights across all turbines in one group."""
    weights = np.zeros(len(grid_coordinates), dtype=float)
    for latitude, longitude in turbine_coordinates:
        dy = grid_coordinates[:, 0] - latitude
        dx = (grid_coordinates[:, 1] - longitude) * np.cos(np.deg2rad(latitude))
        distance_km = 111.0 * np.hypot(dx, dy)
        nearest = np.argsort(distance_km)[:neighbors]
        local = 1.0 / np.maximum(distance_km[nearest], 1e-3) ** 2
        local /= local.sum()
        weights[nearest] += local / len(turbine_coordinates)
    assert np.isclose(weights.sum(), 1.0)
    return weights


def row_weighted(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    denominator = finite @ weights
    assert (denominator > 0).all()
    return np.nansum(values * weights[None, :], axis=1) / denominator


def row_weighted_std(
    values: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(values)
    denominator = finite @ weights
    squared = (values - means[:, None]) ** 2
    variance = np.nansum(squared * weights[None, :], axis=1) / denominator
    return np.sqrt(np.maximum(variance, 0.0))


def load_wind_fields(split: str) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Load raw wind vectors and add compact group-local IDW summaries."""
    merged = None
    raw_columns: dict[str, list[str]] = {}
    metadata = {}

    for source, spec in SOURCE_SPECS.items():
        path = RAW / split / spec["file"].format(split=split)
        columns = [
            "forecast_kst_dtm",
            "grid_id",
            "latitude",
            "longitude",
            spec["u"],
            spec["v"],
        ]
        data = pd.read_csv(
            path,
            encoding="utf-8-sig",
            parse_dates=["forecast_kst_dtm"],
            usecols=columns,
        )
        coordinates = (
            data[["grid_id", "latitude", "longitude"]]
            .drop_duplicates()
            .sort_values("grid_id")
        )
        assert coordinates["grid_id"].is_monotonic_increasing
        grid_ids = coordinates["grid_id"].to_numpy(dtype=int)
        grid_coordinates = coordinates[["latitude", "longitude"]].to_numpy()

        pivots = {}
        source_frame = None
        for component in ("u", "v"):
            pivot = data.pivot(
                index="forecast_kst_dtm",
                columns="grid_id",
                values=spec[component],
            ).reindex(columns=grid_ids)
            pivot.columns = [f"raw_{source}_{component}_{grid}" for grid in grid_ids]
            pivots[component] = pivot
            source_frame = pivot if source_frame is None else source_frame.join(pivot)

        raw_columns[source] = source_frame.columns.tolist()
        source_frame = source_frame.reset_index().rename(
            columns={"forecast_kst_dtm": "kst_dtm"}
        )
        merged = source_frame if merged is None else merged.merge(
            source_frame, on="kst_dtm", how="inner", validate="one_to_one"
        )

        u_values = pivots["u"].to_numpy(dtype=float)
        v_values = pivots["v"].to_numpy(dtype=float)
        speed_values = np.hypot(u_values, v_values)
        source_metadata = {}
        for group in V.GROUPS:
            turbines = decimal_coordinates(TURBINE_DMS[group])
            weights = site_weights(grid_coordinates, turbines)
            mean_u = row_weighted(u_values, weights)
            mean_v = row_weighted(v_values, weights)
            mean_speed = row_weighted(speed_values, weights)
            speed_std = row_weighted_std(speed_values, weights, mean_speed)
            merged[f"site_g{group}_{source}_u"] = mean_u
            merged[f"site_g{group}_{source}_v"] = mean_v
            merged[f"site_g{group}_{source}_ws"] = mean_speed
            merged[f"site_g{group}_{source}_ws_std"] = speed_std
            source_metadata[str(group)] = {
                "centroid": turbines.mean(axis=0).tolist(),
                "nonzero_grid_ids": grid_ids[weights > 0].tolist(),
                "weights": weights[weights > 0].tolist(),
            }
        metadata[source] = source_metadata

    assert merged is not None
    assert merged["kst_dtm"].is_unique
    return merged.sort_values("kst_dtm").reset_index(drop=True), raw_columns, metadata


def add_raw_fields(frame: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(fields, on="kst_dtm", how="left", validate="one_to_one")
    raw = [column for column in fields if column.startswith("raw_")]
    assert out[raw].notna().any(axis=1).all(), "no raw NWP values for a timestamp"
    return out


def site_columns(group: int) -> list[str]:
    return [
        f"site_g{group}_{source}_{suffix}"
        for source in SOURCE_SPECS
        for suffix in ("u", "v", "ws", "ws_std")
    ]


def append_fold_pca(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    raw_columns: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    validation = validation.copy()
    output_columns = []
    for source, columns in raw_columns.items():
        transformer = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=PCA_COMPONENTS, random_state=42),
        )
        train_scores = transformer.fit_transform(train[columns])
        validation_scores = transformer.transform(validation[columns])
        for component in range(PCA_COMPONENTS):
            name = f"fold_{source}_pc{component + 1}"
            train[name] = train_scores[:, component]
            validation[name] = validation_scores[:, component]
            output_columns.append(name)
    return train, validation, output_columns


def prepare_specific_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    group: int,
    mode: str,
    fields: pd.DataFrame,
    raw_columns: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = add_raw_fields(train, fields)
    validation = add_raw_fields(validation, fields)
    features = list(V.FEATS)
    if mode in {"site", "site_pca"}:
        features.extend(site_columns(group))
    if mode in {"pca", "site_pca"}:
        train, validation, pca_columns = append_fold_pca(
            train, validation, raw_columns
        )
        features.extend(pca_columns)
    if mode not in {"base", "site", "pca", "site_pca"}:
        raise ValueError(f"unknown specific mode: {mode}")
    return train, validation, features


def fit_specific(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    group: int,
    mode: str,
    fields: pd.DataFrame,
    raw_columns: dict[str, list[str]],
) -> np.ndarray:
    train, validation, features = prepare_specific_frames(
        train, validation, group, mode, fields, raw_columns
    )
    model = lgb.LGBMRegressor(**V.GBM_PARAMS).fit(
        train[features],
        train[V.TGT[group]],
        sample_weight=M.metric_gbm_weight(train, group),
    )
    return np.clip(model.predict(validation[features]), 0, W.CAP[group])


def fit_pooled_group3(
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> np.ndarray:
    """Fit a capacity-normalized pooled GBM and predict only group 3."""
    pooled = []
    for group, (train, _) in frames.items():
        part = train[V.FEATS].copy()
        part["group_id"] = group - 1
        part["target_cf"] = train[V.TGT[group]].to_numpy() / W.CAP[group]
        part["sample_weight"] = M.metric_gbm_weight(train, group)
        pooled.append(part)
    pooled_train = pd.concat(pooled, ignore_index=True)
    features = list(V.FEATS) + ["group_id"]
    model = lgb.LGBMRegressor(**V.GBM_PARAMS).fit(
        pooled_train[features],
        pooled_train["target_cf"],
        sample_weight=pooled_train["sample_weight"],
    )
    validation = frames[3][1][V.FEATS].copy()
    validation["group_id"] = 2
    return np.clip(model.predict(validation[features]), 0, 1) * W.CAP[3]


def score(actual: np.ndarray, prediction: np.ndarray, group: int) -> dict[str, float]:
    return M.group_result(actual, prediction, W.CAP[group])


def annual_frames() -> dict[int, dict[int, tuple[pd.DataFrame, pd.DataFrame]]]:
    return {2023: V.make_2023_frames(), 2024: V.make_2024_frames()}


def cutoff_frames(
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    frames = {}
    for group in V.GROUPS:
        source = V.FR[group]
        train = source[source["kst_dtm"] < validation_start]
        validation = source[
            (source["kst_dtm"] >= validation_start)
            & (source["kst_dtm"] < validation_end)
        ]
        if train.empty or validation.empty:
            continue
        iso = W.fit_powercurve(train, V.TGT[group], W.CAP[group])
        frames[group] = (W.with_pc(train, iso), W.with_pc(validation, iso))
    return frames


def monthly_scores(
    validation: pd.DataFrame,
    base_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    group: int,
) -> dict[str, dict[str, float]]:
    output = {}
    month_key = validation["kst_dtm"].dt.to_period("M").astype(str)
    actual = validation[V.TGT[group]].to_numpy()
    for month in sorted(month_key.unique()):
        keep = month_key.to_numpy() == month
        base = score(actual[keep], base_prediction[keep], group)
        candidate = score(actual[keep], candidate_prediction[keep], group)
        output[month] = {
            "base": base["score"],
            "candidate": candidate["score"],
            "delta": candidate["score"] - base["score"],
            "valid_rows": int((actual[keep] >= W.VALID_CF * W.CAP[group]).sum()),
        }
    return output


def choose_annual_candidate(
    annual_results: dict,
    group: int,
    candidates: list[str],
) -> str:
    eligible = []
    years = [year for year, groups in annual_results.items() if str(group) in groups]
    for candidate in candidates:
        deltas = [
            annual_results[year][str(group)][candidate]["score"]
            - annual_results[year][str(group)]["base"]["score"]
            for year in years
        ]
        if deltas and min(deltas) >= 0.0 and np.mean(deltas) >= MIN_ANNUAL_DELTA:
            eligible.append((float(np.mean(deltas)), candidate))
    return max(eligible, default=(0.0, "base"))[1]


def adoption_decision(
    annual_results: dict,
    quarterly_results: dict,
    monthly_results: dict,
    group: int,
    candidate: str,
) -> dict:
    if candidate == "base":
        return {"adopt": False, "reason": "no annual-safe candidate"}

    years = [year for year, groups in annual_results.items() if str(group) in groups]
    annual_deltas = [
        annual_results[year][str(group)][candidate]["score"]
        - annual_results[year][str(group)]["base"]["score"]
        for year in years
    ]
    quarters = quarterly_results[str(group)]
    quarter_deltas = [entry["delta"] for entry in quarters.values()]
    quarter_year_means = {}
    for year in sorted({period[:4] for period in quarters}):
        values = [entry["delta"] for period, entry in quarters.items() if period[:4] == year]
        quarter_year_means[year] = float(np.mean(values))
    months = monthly_results[str(group)]
    month_deltas = [entry["delta"] for entry in months.values()]

    checks = {
        "annual_nonnegative": min(annual_deltas) >= 0.0,
        "annual_mean_material": float(np.mean(annual_deltas)) >= MIN_ANNUAL_DELTA,
        "quarter_mean_positive": float(np.mean(quarter_deltas)) > 0.0,
        "quarter_positive_fraction": float(np.mean(np.asarray(quarter_deltas) >= 0.0))
        >= MIN_POSITIVE_FRACTION,
        "each_quarter_year_nonnegative": min(quarter_year_means.values()) >= 0.0,
        "month_median_positive": float(np.median(month_deltas)) > 0.0,
        "month_positive_fraction": float(np.mean(np.asarray(month_deltas) >= 0.0))
        >= 0.50,
    }
    return {
        "adopt": all(checks.values()),
        "candidate": candidate,
        "checks": checks,
        "annual_deltas": annual_deltas,
        "quarter_mean_delta": float(np.mean(quarter_deltas)),
        "quarter_year_mean_delta": quarter_year_means,
        "quarter_positive_fraction": float(np.mean(np.asarray(quarter_deltas) >= 0.0)),
        "month_median_delta": float(np.median(month_deltas)),
        "month_positive_fraction": float(np.mean(np.asarray(month_deltas) >= 0.0)),
    }


def main() -> None:
    fields, raw_columns, spatial_metadata = load_wind_fields("train")
    folds = annual_frames()
    annual_results = {}
    annual_predictions = {}

    for year, frames in folds.items():
        print(f"annual fold {year}")
        annual_results[str(year)] = {}
        for group, (train, validation) in frames.items():
            actual = validation[V.TGT[group]].to_numpy()
            annual_results[str(year)][str(group)] = {}
            for mode in ("base", "site", "pca", "site_pca"):
                prediction = fit_specific(
                    train, validation, group, mode, fields, raw_columns
                )
                annual_predictions[(year, group, mode)] = prediction
                result = score(actual, prediction, group)
                annual_results[str(year)][str(group)][mode] = result
                print(year, group, mode, round(result["score"], 6))
            if group == 3:
                prediction = fit_pooled_group3(frames)
                annual_predictions[(year, group, "pooled")] = prediction
                result = score(actual, prediction, group)
                annual_results[str(year)][str(group)]["pooled"] = result
                print(year, group, "pooled", round(result["score"], 6))

    selected = {}
    for group in V.GROUPS:
        candidates = ["site", "pca", "site_pca"]
        if group == 3:
            candidates.append("pooled")
        selected[group] = choose_annual_candidate(annual_results, group, candidates)
    print("annual selections", selected)

    quarterly_results = {str(group): {} for group in V.GROUPS}
    for group in V.GROUPS:
        if selected[group] == "base":
            continue
        first_period = "2023Q3" if group == 3 else "2023Q1"
        for period in pd.period_range(first_period, "2024Q4", freq="Q"):
            frames = cutoff_frames(period.start_time, (period + 1).start_time)
            train, validation = frames[group]
            base_prediction = fit_specific(
                train, validation, group, "base", fields, raw_columns
            )
            if selected[group] == "pooled":
                candidate_prediction = fit_pooled_group3(frames)
            else:
                candidate_prediction = fit_specific(
                    train,
                    validation,
                    group,
                    selected[group],
                    fields,
                    raw_columns,
                )
            actual = validation[V.TGT[group]].to_numpy()
            base_result = score(actual, base_prediction, group)
            candidate_result = score(actual, candidate_prediction, group)
            quarterly_results[str(group)][str(period)] = {
                "base": base_result["score"],
                "candidate": candidate_result["score"],
                "delta": candidate_result["score"] - base_result["score"],
                "one_minus_nmae_delta": candidate_result["one_minus_nmae"]
                - base_result["one_minus_nmae"],
                "ficr_delta": candidate_result["ficr"] - base_result["ficr"],
            }
            print(
                "quarter",
                group,
                period,
                selected[group],
                round(candidate_result["score"] - base_result["score"], 6),
            )

    monthly_results = {str(group): {} for group in V.GROUPS}
    for year, frames in folds.items():
        for group, (_, validation) in frames.items():
            candidate = selected[group]
            if candidate == "base":
                continue
            monthly_results[str(group)].update(
                monthly_scores(
                    validation,
                    annual_predictions[(year, group, "base")],
                    annual_predictions[(year, group, candidate)],
                    group,
                )
            )

    decisions = {
        str(group): adoption_decision(
            annual_results,
            quarterly_results,
            monthly_results,
            group,
            selected[group],
        )
        for group in V.GROUPS
    }
    print("decisions", decisions)

    result = {
        "recipe": {
            "site_neighbors": SITE_NEIGHBORS,
            "pca_components_per_source": PCA_COMPONENTS,
            "minimum_annual_delta": MIN_ANNUAL_DELTA,
            "minimum_positive_quarter_fraction": MIN_POSITIVE_FRACTION,
            "base_features": len(V.FEATS),
            "site_features": 8,
            "pca_features": 2 * PCA_COMPONENTS,
        },
        "spatial_metadata": spatial_metadata,
        "annual": annual_results,
        "selected_after_annual": {str(group): mode for group, mode in selected.items()},
        "quarterly": quarterly_results,
        "monthly_holdout": monthly_results,
        "decisions": decisions,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
