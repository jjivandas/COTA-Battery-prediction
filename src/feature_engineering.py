"""Feature engineering for battery degradation prediction."""

import pandas as pd
import numpy as np
from src.config import (
    MASTER_DATASET_PATH, FEATURE_MATRIX_PATH, PROCESSED_DIR,
    CUMULATIVE_COLS, WEEKLY_COLS, TARGET_CUMULATIVE, TARGET_DELTA,
)


def compute_delta_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bus week-over-week deltas for cumulative targets."""
    df = df.sort_values(["bus_id", "week_index"]).copy()
    for col in ["avg_qloss", "avg_qloss_cycling", "avg_qloss_calendar"]:
        delta_col = col.replace("avg_", "delta_")
        df[delta_col] = df.groupby("bus_id")[col].diff()
    return df


def compute_soc_range(df: pd.DataFrame) -> pd.DataFrame:
    """SOC range = max_soc - min_soc."""
    df["soc_range"] = df["max_soc"] - df["min_soc"]
    return df


def compute_regen_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Regen ratio = weekly regen energy / weekly net energy."""
    df["regen_ratio"] = np.where(
        df["weekly_net_energy"] > 0,
        df["weekly_regen_energy"] / df["weekly_net_energy"],
        0.0,
    )
    return df


def compute_utilization_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Driving, charging, parking as fraction of total time."""
    total_time = (
        df["weekly_driving_time"]
        + df["weekly_charging_time"]
        + df["weekly_parking_time"]
    )
    for col, name in [
        ("weekly_driving_time", "driving_frac"),
        ("weekly_charging_time", "charging_frac"),
        ("weekly_parking_time", "parking_frac"),
    ]:
        df[name] = np.where(total_time > 0, df[col] / total_time, 0.0)
    return df


def compute_temp_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Temperature delta = battery temp - ambient temp."""
    df["temp_delta"] = df["avg_batt_temp"] - df["avg_amb_temp"]
    return df


def compute_route_count(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct routes per week (comma-separated in 'routes' column)."""
    df["route_count"] = df["routes"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() else 0
    )
    return df


def compute_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """1-week lag within each bus for key features."""
    df = df.sort_values(["bus_id", "week_index"])
    for col in ["delta_qloss", "avg_batt_temp"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("bus_id")[col].shift(1)
    return df


def compute_rolling_features(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Rolling mean (4-week window) within each bus for key features."""
    df = df.sort_values(["bus_id", "week_index"])
    for col in ["weekly_distance", "avg_batt_temp"]:
        df[f"{col}_roll{window}"] = (
            df.groupby("bus_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    return df


def build_feature_matrix(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    if df is None:
        print(f"Loading master dataset from {MASTER_DATASET_PATH}")
        df = pd.read_csv(MASTER_DATASET_PATH, parse_dates=["week_start", "week_end"])

    print("Computing features...")
    df = compute_delta_targets(df)
    df = compute_soc_range(df)
    df = compute_regen_ratio(df)
    df = compute_utilization_fractions(df)
    df = compute_temp_delta(df)
    df = compute_route_count(df)
    df = compute_lagged_features(df)
    df = compute_rolling_features(df)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_MATRIX_PATH, index=False)
    print(f"Saved feature matrix to {FEATURE_MATRIX_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nNew feature columns:")
    new_cols = [
        "delta_qloss", "delta_qloss_cycling", "delta_qloss_calendar",
        "soc_range", "regen_ratio", "driving_frac", "charging_frac",
        "parking_frac", "temp_delta", "route_count",
        "delta_qloss_lag1", "avg_batt_temp_lag1",
        "weekly_distance_roll4", "avg_batt_temp_roll4",
    ]
    for c in new_cols:
        if c in df.columns:
            print(f"  {c}: {df[c].notna().sum()} non-null, "
                  f"mean={df[c].mean():.4f}, std={df[c].std():.4f}")

    return df


def get_feature_sets(df: pd.DataFrame) -> dict:
    """Return feature column lists for cumulative and delta models."""
    engineered = [
        "soc_range", "regen_ratio", "driving_frac", "charging_frac",
        "parking_frac", "temp_delta", "route_count",
        "delta_qloss_lag1", "avg_batt_temp_lag1",
        "weekly_distance_roll4", "avg_batt_temp_roll4",
    ]

    cumulative_features = ["week_index"] + CUMULATIVE_COLS[:-3] + WEEKLY_COLS + engineered
    # Remove target-related cumulative cols to avoid leakage
    cumulative_features = [c for c in cumulative_features if c != TARGET_CUMULATIVE]

    delta_features = ["week_index"] + WEEKLY_COLS + engineered

    return {
        "cumulative": cumulative_features,
        "delta": delta_features,
    }


if __name__ == "__main__":
    build_feature_matrix()
