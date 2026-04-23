"""Noise injection utilities for simulated weekly aggregates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.config import (
    COLUMN_NOISE,
    CUMULATIVE_RECOMPUTE_MAP,
    NOISE_ENABLED,
    NOISE_PROFILE_NAME,
    NOISE_RANDOM_SEED,
    NOISE_SERVICE_WEEKS_ONLY,
)

WEEKLY_TIME_COLS = [
    "weekly_driving_time",
    "weekly_charging_time",
    "weekly_parking_time",
]


def get_noise_metadata() -> dict:
    """Return a serializable summary of the active noise profile."""
    return {
        "enabled": NOISE_ENABLED,
        "profile_name": NOISE_PROFILE_NAME,
        "random_seed": NOISE_RANDOM_SEED,
        "service_weeks_only": NOISE_SERVICE_WEEKS_ONLY,
        "column_noise": COLUMN_NOISE,
        "recomputed_cumulative_columns": CUMULATIVE_RECOMPUTE_MAP,
    }


def apply_noise_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Perturb weekly inputs with column-specific noise rules."""
    if not NOISE_ENABLED:
        return df

    baseline = df.copy()
    noisy = df.copy()
    rng = np.random.default_rng(NOISE_RANDOM_SEED)

    if NOISE_SERVICE_WEEKS_ONLY and "is_service_week" in noisy.columns:
        base_mask = noisy["is_service_week"].fillna(False).astype(bool)
    else:
        base_mask = pd.Series(True, index=noisy.index)

    for col, spec in COLUMN_NOISE.items():
        if col not in noisy.columns:
            continue
        col_mask = base_mask & noisy[col].notna()
        if not col_mask.any():
            continue
        noisy.loc[col_mask, col] = _apply_series_noise(noisy.loc[col_mask, col], spec, rng)

    noisy = _repair_soc_columns(noisy, base_mask)
    noisy = _normalize_weekly_time_budget(noisy, baseline, base_mask)
    noisy = _recompute_cumulative_columns(noisy)
    return noisy


def _apply_series_noise(
    series: pd.Series,
    spec: Mapping[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply additive or relative Gaussian noise to a numeric series."""
    values = series.to_numpy(dtype=float, copy=True)
    std = float(spec["std"])
    strategy = spec["strategy"]

    noise = rng.normal(loc=0.0, scale=std, size=len(values))
    if strategy == "additive":
        values = values + noise
    elif strategy == "relative":
        values = values * (1.0 + noise)
    else:
        raise ValueError(f"Unsupported noise strategy: {strategy}")

    clip_bounds = spec.get("clip")
    if clip_bounds is not None:
        lower, upper = clip_bounds
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        values = np.clip(values, lower, upper)

    return values


def _repair_soc_columns(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Keep min/avg/max SOC ordered after perturbation."""
    needed = ["min_soc", "avg_soc", "max_soc"]
    if not all(col in df.columns for col in needed):
        return df

    valid = mask & df[needed].notna().all(axis=1)
    if not valid.any():
        return df

    values = df.loc[valid, needed].to_numpy(dtype=float, copy=True)
    values.sort(axis=1)
    df.loc[valid, "min_soc"] = values[:, 0]
    df.loc[valid, "avg_soc"] = values[:, 1]
    df.loc[valid, "max_soc"] = values[:, 2]
    return df


def _normalize_weekly_time_budget(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    mask: pd.Series,
) -> pd.DataFrame:
    """Preserve each week's total time after perturbing time subcomponents."""
    if not all(col in df.columns for col in WEEKLY_TIME_COLS):
        return df

    valid = mask & df[WEEKLY_TIME_COLS].notna().all(axis=1)
    if not valid.any():
        return df

    times = df.loc[valid, WEEKLY_TIME_COLS].to_numpy(dtype=float, copy=True)
    original_total = baseline.loc[valid, WEEKLY_TIME_COLS].to_numpy(dtype=float, copy=True).sum(axis=1)
    noisy_total = np.maximum(times.sum(axis=1), 1e-12)
    scaled = times * (original_total / noisy_total)[:, None]
    df.loc[valid, WEEKLY_TIME_COLS] = scaled
    return df


def _recompute_cumulative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild cumulative totals from their weekly counterparts after noise."""
    if "bus_id" not in df.columns:
        return df

    ordered = df.sort_values(["bus_id", "week_index"]).copy()
    for weekly_col, cumulative_col in CUMULATIVE_RECOMPUTE_MAP.items():
        if weekly_col not in ordered.columns or cumulative_col not in ordered.columns:
            continue
        ordered[cumulative_col] = ordered.groupby("bus_id")[weekly_col].cumsum()

    return ordered.sort_index()
