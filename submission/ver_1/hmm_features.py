"""Rejected NWP-only HMM regime features retained for experiment replay."""

from __future__ import annotations

import pandas as pd


EPS = 1e-6
REGIME_VARS = [
    "hub_v",
    "shear_gfs",
    "alpha_gfs",
    "gust_ratio",
    "air_density",
    "gfs_ldaps_diff",
    "gfs_wind_speed_850hpa_mean",
]


def fit_regime_hmm(fr_train, n_states=4, seed=42, cols=None):
    """Fit a Gaussian HMM using only training-period NWP features."""
    from hmmlearn.hmm import GaussianHMM

    cols = cols or REGIME_VARS
    values = fr_train[cols].to_numpy()
    mean = values.mean(0)
    std = values.std(0) + EPS
    scaled = (values - mean) / std
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=seed,
        tol=1e-3,
    )
    model.fit(scaled)
    return (mean, std, cols), model


def regime_posteriors(fr, scaler, model):
    """Return soft regime posteriors for a fitted HMM."""
    mean, std, cols = scaler
    scaled = (fr[cols].to_numpy() - mean) / std
    posterior = model.predict_proba(scaled)
    return pd.DataFrame(
        posterior,
        columns=[f"regime_{index}" for index in range(posterior.shape[1])],
        index=fr.index,
    )
