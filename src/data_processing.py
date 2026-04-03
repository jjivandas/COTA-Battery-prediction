"""Load, clean, and merge the per-bus CSV data."""

import pandas as pd
import numpy as np
from src.config import (
    BUS_IDS, get_csv_path, COLUMN_RENAME, DATE_COLUMNS, DATE_FORMAT,
    CUMULATIVE_COLS, PROCESSED_DIR, MASTER_DATASET_PATH,
)


def load_single_bus(bus_id: str) -> pd.DataFrame:
    """Load and clean a single bus CSV."""
    path = get_csv_path(bus_id)
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAME)
    df["bus_id"] = int(bus_id)

    # Parse dates
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT)

    return df


def load_all_buses() -> pd.DataFrame:
    """Load all 12 bus CSVs, concatenate, and sort."""
    frames = []
    for bus_id in BUS_IDS:
        df = load_single_bus(bus_id)
        frames.append(df)
        print(f"  Bus {bus_id}: {len(df)} rows, weeks {df['week_index'].min()}-{df['week_index'].max()}")

    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values(["bus_id", "week_index"]).reset_index(drop=True)
    return master


def flag_service_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Flag weeks with no service (zero distance)."""
    df["is_service_week"] = df["weekly_distance"] > 0
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill cumulative columns within each bus group.
    Leave target NaNs as-is for honest evaluation."""
    cum_cols_present = [c for c in CUMULATIVE_COLS if c in df.columns]
    df[cum_cols_present] = df.groupby("bus_id")[cum_cols_present].ffill()
    return df


def build_master_dataset() -> pd.DataFrame:
    """Full data processing pipeline."""
    print("Loading all buses...")
    df = load_all_buses()

    print("\nFlagging service weeks...")
    df = flag_service_weeks(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_DATASET_PATH, index=False)
    print(f"\nSaved master dataset to {MASTER_DATASET_PATH}")

    # Quality summary
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Total buses: {df['bus_id'].nunique()}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nRows per bus:")
    print(df.groupby("bus_id").size().to_string())
    print(f"\nDate range: {df['week_start'].min()} to {df['week_end'].max()}")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("  None")
    print(f"\nService weeks: {df['is_service_week'].sum()} / {len(df)}")
    print(f"Non-service weeks: {(~df['is_service_week']).sum()}")
    print(f"\nTarget (avg_qloss) range: {df['avg_qloss'].min():.4f} - {df['avg_qloss'].max():.4f}")

    return df


if __name__ == "__main__":
    build_master_dataset()
