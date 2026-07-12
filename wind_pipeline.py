"""Shared, lazily loaded model foundation for submission versions 10 and later."""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

import wind_lib as W
from wind_paths import raw_data_dir


torch.set_num_threads(1)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEV = DEVICE  # Historical compatibility.
GROUPS = (1, 2, 3)
STUCK_WEIGHT = 0.5
STUCK_W = STUCK_WEIGHT  # Historical compatibility.
STUCK_THRESHOLD = 0.3
EARLY_STOP_TRAIN_FRACTION = 0.85

MLP_CFG = {
    "h": 256,
    "depth": 3,
    "drop": 0.0,
    "lr": 0.0015868563457489381,
    "wd": 1e-4,
    "bs": 256,
    "emb": 4,
}

GBM_PARAMS = {
    "objective": "mae",
    "n_estimators": 2000,
    "learning_rate": 0.020651822836313095,
    "num_leaves": 63,
    "min_child_samples": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


@dataclass
class PipelineContext:
    """Prepared train/test frames and their stable feature contract."""

    train_frames: dict[int, pd.DataFrame]
    targets: dict[int, str]
    test_frames: dict[int, pd.DataFrame]
    features: list[str]


class MLP(nn.Module):
    def __init__(
        self,
        feature_count: int,
        group_count: int = 3,
        hidden_size: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
        embedding_size: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(group_count, embedding_size)
        layers: list[nn.Module] = [
            nn.Linear(feature_count + embedding_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout)]
            )
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(groups)
        return self.network(torch.cat([values, embedded], -1)).squeeze(-1)


def histgbm() -> HGB:
    return HGB(
        loss="absolute_error",
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        random_state=42,
    )


def load_frames() -> tuple[
    dict[int, pd.DataFrame],
    dict[int, str],
    dict[int, pd.DataFrame],
    list[str],
]:
    train_frames: dict[int, pd.DataFrame] = {}
    targets: dict[int, str] = {}
    test_frames: dict[int, pd.DataFrame] = {}
    for group in GROUPS:
        raw_train, target = W.load_train(group)
        targets[group] = target
        train_frames[group] = W.add_spatial(W.build(raw_train, group), "train")
        test_frames[group] = W.add_spatial(
            W.build(W.load_test(group), group), "test"
        )
    base = [
        column
        for column in W.feature_cols(train_frames[1])
        if column not in W.SPATIAL_COLS
    ] + ["pc_pred_cf"]
    features = W.lean_features(base) + W.SPATIAL_COLS
    if len(features) != len(set(features)):
        raise ValueError("feature contract contains duplicate columns")
    return train_frames, targets, test_frames, features


def load_stuck_fractions(raw=None) -> dict[int, pd.Series]:
    """Load SCADA-derived stuck fractions by group and hourly timestamp."""
    raw = raw_data_dir() if raw is None else raw
    frames = {}
    for filename, prefix, turbine_count, rated_power in [
        ("scada_vestas_train.csv", "vestas", 12, 3600.0),
        ("scada_unison_train.csv", "unison", 5, 4200.0),
    ]:
        frame = pd.read_csv(
            raw / "train" / filename,
            encoding="utf-8-sig",
            parse_dates=["kst_dtm"],
        )
        frame["hour"] = frame["kst_dtm"].dt.ceil("h")
        turbines = []
        for index in range(1, turbine_count + 1):
            power = f"{prefix}_wtg{index:02d}_power_kw10m"
            wind = f"{prefix}_wtg{index:02d}_ws"
            hourly = frame.groupby("hour").agg(
                power_mean=(power, "mean"),
                wind_mean=(wind, "mean"),
            )
            turbines.append(
                pd.DataFrame(
                    {
                        f"stuck_{index}": (
                            (hourly.wind_mean >= 4.0)
                            & (hourly.power_mean <= 0.01 * rated_power)
                        ).astype(float),
                        f"reported_{index}": hourly.power_mean.notna().astype(float),
                    }
                )
            )
        frames[prefix] = pd.concat(turbines, axis=1)

    def fraction(prefix: str, ids: range) -> pd.Series:
        frame = frames[prefix]
        stuck = frame[[f"stuck_{index}" for index in ids]].sum(axis=1)
        reported = frame[[f"reported_{index}" for index in ids]].sum(axis=1)
        return (stuck / reported).where(reported >= 3)

    return {
        1: fraction("vestas", range(1, 7)),
        2: fraction("vestas", range(7, 13)),
        3: fraction("unison", range(1, 6)),
    }


_CONTEXT: PipelineContext | None = None


def context() -> PipelineContext:
    """Load and cache prepared frames on first use, never during module import."""
    global _CONTEXT
    if _CONTEXT is None:
        train_frames, targets, test_frames, features = load_frames()
        stuck_fractions = load_stuck_fractions()
        for group in GROUPS:
            values = stuck_fractions[group].reindex(
                train_frames[group].kst_dtm
            ).to_numpy()
            train_frames[group] = train_frames[group].assign(
                stuck_mask=np.nan_to_num(values, nan=0.0) >= STUCK_THRESHOLD
            )
        _CONTEXT = PipelineContext(
            train_frames=train_frames,
            targets=targets,
            test_frames=test_frames,
            features=features,
        )
    return _CONTEXT


def clear_context() -> None:
    """Clear the cached context for tests or a changed data configuration."""
    global _CONTEXT
    _CONTEXT = None


def __getattr__(name: str):
    legacy_fields = {
        "FR": "train_frames",
        "TGT": "targets",
        "FR_TE": "test_frames",
        "FEATS": "features",
    }
    if name in legacy_fields:
        return getattr(context(), legacy_fields[name])
    raise AttributeError(name)


def train_one(pool_train: pd.DataFrame, seed: int):
    """Train one pooled MLP with a chronological early-stopping tail."""
    features = context().features
    torch.manual_seed(seed)
    np.random.seed(seed)
    pool_train = pool_train.sort_values("kst_dtm")
    mean = pool_train[features].mean()
    std = pool_train[features].std() + 1e-8
    values = ((pool_train[features] - mean) / std).to_numpy(np.float32)
    targets = pool_train["cf"].to_numpy(np.float32)
    groups = pool_train["gid"].to_numpy(np.int64)
    weights = pool_train["w"].to_numpy(np.float32)
    cut = int(len(values) * EARLY_STOP_TRAIN_FRACTION)
    if cut <= 0 or cut >= len(values):
        raise ValueError("MLP training requires non-empty train and validation slices")

    value_tensor = torch.tensor(values, device=DEVICE)
    target_tensor = torch.tensor(targets, device=DEVICE)
    group_tensor = torch.tensor(groups, device=DEVICE)
    weight_tensor = torch.tensor(weights, device=DEVICE)

    model = MLP(
        len(features),
        hidden_size=MLP_CFG["h"],
        depth=MLP_CFG["depth"],
        dropout=MLP_CFG["drop"],
        embedding_size=MLP_CFG["emb"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_CFG["lr"],
        weight_decay=MLP_CFG["wd"],
    )
    best_loss = float("inf")
    best_state = None
    patience = 0
    for _ in range(120):
        model.train()
        permutation = np.random.permutation(np.arange(cut))
        for start in range(0, len(permutation), MLP_CFG["bs"]):
            batch = torch.tensor(
                permutation[start : start + MLP_CFG["bs"]], device=DEVICE
            )
            optimizer.zero_grad()
            error = (
                model(value_tensor[batch], group_tensor[batch])
                - target_tensor[batch]
            ).abs()
            weighted_loss = (error * weight_tensor[batch]).sum() / (
                weight_tensor[batch].sum() + 1e-8
            )
            weighted_loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            error = (
                model(value_tensor[cut:], group_tensor[cut:])
                - target_tensor[cut:]
            ).abs()
            validation_loss = (
                (error * weight_tensor[cut:]).sum()
                / (weight_tensor[cut:].sum() + 1e-8)
            ).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            break
    if best_state is None:
        raise RuntimeError("MLP early stopping did not produce a valid state")
    model.load_state_dict(best_state)
    return model, (mean, std)


def predict_one(
    model: MLP,
    scaler,
    frame: pd.DataFrame,
    group: int,
) -> np.ndarray:
    features = context().features
    mean, std = scaler
    values = torch.tensor(
        ((frame[features] - mean) / std).to_numpy(np.float32),
        device=DEVICE,
    )
    groups = torch.full(
        (len(frame),), group - 1, dtype=torch.long, device=DEVICE
    )
    model.eval()
    with torch.no_grad():
        prediction = model(values, groups).cpu().numpy()
    return np.clip(prediction, 0, 1) * W.CAP[group]


def make_2023_frames() -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    prepared = context()
    frames = {}
    for group in (1, 2):
        frame = prepared.train_frames[group]
        years = frame.kst_dtm.dt.year
        train = frame[years == 2022]
        validation = frame[years == 2023]
        curve = W.fit_powercurve(
            train, prepared.targets[group], W.CAP[group]
        )
        frames[group] = (W.with_pc(train, curve), W.with_pc(validation, curve))
    return frames


def make_2024_frames() -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    prepared = context()
    frames = {}
    for group in GROUPS:
        frame = prepared.train_frames[group]
        train = frame[frame.kst_dtm < W.VALID_START]
        validation = frame[frame.kst_dtm >= W.VALID_START]
        curve = W.fit_powercurve(
            train, prepared.targets[group], W.CAP[group]
        )
        frames[group] = (W.with_pc(train, curve), W.with_pc(validation, curve))
    return frames
