"""Configuration: paths, column mappings, constants."""

import os
from pathlib import Path

# ── Dataset selection ──────────────────────────────────────────────────────
# BASE_DATASET points at the raw simulation folder under data/raw/.
# DATASET_VARIANT controls which processed/results namespace to use.
# Examples:
#   clean    -> data/processed/mar-2026, results/mar-2026
#   noisy-v1 -> data/processed/mar-2026-noisy-v1, results/mar-2026-noisy-v1
BASE_DATASET = os.getenv("BASE_DATASET", "mar-2026")
DATASET_VARIANT = os.getenv("DATASET_VARIANT", "clean")
IS_NOISY_VARIANT = DATASET_VARIANT != "clean"
ACTIVE_DATASET = (
    BASE_DATASET if DATASET_VARIANT == "clean"
    else f"{BASE_DATASET}-{DATASET_VARIANT}"
)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / BASE_DATASET / "per-bus-csvs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / ACTIVE_DATASET
RESULTS_DIR = PROJECT_ROOT / "results" / ACTIVE_DATASET
FIGURES_DIR = RESULTS_DIR / "figures"

MASTER_DATASET_PATH = PROCESSED_DIR / "master_dataset.csv"
FEATURE_MATRIX_PATH = PROCESSED_DIR / "feature_matrix.csv"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
DATASET_METADATA_PATH = PROCESSED_DIR / "dataset_config.json"
NOISE_METADATA_PATH = PROCESSED_DIR / "noise_config.json"

# ── Bus IDs ────────────────────────────────────────────────────────────────
BUS_IDS = [f"{i:03d}" for i in range(1, 13)]
NUM_BUSES = 12

# Bus-002 has a different filename
def get_csv_path(bus_id: str) -> Path:
    """Return the CSV path for a given bus ID string (e.g. '001')."""
    bus_dir = RAW_DATA_DIR / f"Bus-Id-{bus_id}"
    if bus_id == "002":
        return bus_dir / "sim_weekly_agg_002.csv"
    return bus_dir / "sim_weekly_agg.csv"

# ── Column rename mapping ─────────────────────────────────────────────────
# Original name -> clean snake_case name (units stripped)
COLUMN_RENAME = {
    "Week_Index": "week_index",
    "Week_Start": "week_start",
    "Week_End": "week_end",
    "Routes": "routes",
    "Avg_SOC": "avg_soc",
    "Max_SOC [-]": "max_soc",
    "Min_SOC [-]": "min_soc",
    "Avg_DOD [-]": "avg_dod",
    "Avg_Batt_Temp": "avg_batt_temp",
    "Avg_Amb_Temp [degC]": "avg_amb_temp",
    "weeklyAvgBattP": "weekly_avg_batt_power",
    "weeklyDistance [km]": "weekly_distance",
    "weeklyTotDistance [km]": "weekly_tot_distance",
    "weeklyDRegEn [kWh]": "weekly_regen_energy",
    "weeklyDNetEn [kWh]": "weekly_net_energy",
    "weeklyEnAux [kWh]": "weekly_aux_energy",
    "weeklyTotRegEn [kWh]": "weekly_tot_regen_energy",
    "weeklyTotNetEn [kWh]": "weekly_tot_net_energy",
    "weeklyTotAux [kWh]": "weekly_tot_aux_energy",
    "weeklyAvgCrateChg [1/h]": "weekly_avg_crate_chg",
    "weeklyCycles [-]": "weekly_cycles",
    "weeklyTotCycles [-]": "weekly_tot_cycles",
    "weeklyDrivingTime [h]": "weekly_driving_time",
    "weeklyChargingTime [h]": "weekly_charging_time",
    "weeklyparkingTime [h]": "weekly_parking_time",
    "Avg_Qcyc": "avg_qloss_cycling",
    "Avg_Qcal": "avg_qloss_calendar",
    "Avg_Qloss": "avg_qloss",
}

# ── Column groups ──────────────────────────────────────────────────────────
DATE_COLUMNS = ["week_start", "week_end"]
DATE_FORMAT = "%d-%b-%Y"

# Cumulative columns (monotonically increasing within a bus)
CUMULATIVE_COLS = [
    "weekly_tot_distance",
    "weekly_tot_regen_energy",
    "weekly_tot_net_energy",
    "weekly_tot_aux_energy",
    "weekly_tot_cycles",
    "avg_qloss_cycling",
    "avg_qloss_calendar",
    "avg_qloss",
]

# Weekly (per-period) columns
WEEKLY_COLS = [
    "avg_soc", "max_soc", "min_soc", "avg_dod",
    "avg_batt_temp", "avg_amb_temp",
    "weekly_avg_batt_power", "weekly_distance",
    "weekly_regen_energy", "weekly_net_energy", "weekly_aux_energy",
    "weekly_avg_crate_chg", "weekly_cycles",
    "weekly_driving_time", "weekly_charging_time", "weekly_parking_time",
]

# Target columns
TARGET_CUMULATIVE = "avg_qloss"
TARGET_DELTA = "delta_qloss"

# ── Optional noise injection for simulated inputs ─────────────────────────
# The noisy dataset variant perturbs measured weekly inputs only, then
# recomputes cumulative totals so the dataset stays internally consistent.
NOISE_ENABLED = IS_NOISY_VARIANT
NOISE_PROFILE_NAME = "default_sensor_noise_v1"
NOISE_RANDOM_SEED = 42
NOISE_SERVICE_WEEKS_ONLY = True

# Columns that can safely absorb measurement noise before feature engineering.
# Strategy:
# - additive: x + N(0, std)
# - relative: x * (1 + N(0, std))
COLUMN_NOISE = {
    "avg_soc": {"strategy": "additive", "std": 0.003, "clip": (0.0, 1.0)},
    "max_soc": {"strategy": "additive", "std": 0.0005, "clip": (0.0, 1.0)},
    "min_soc": {"strategy": "additive", "std": 0.006, "clip": (0.0, 1.0)},
    "avg_dod": {"strategy": "additive", "std": 0.01, "clip": (0.0, 1.0)},
    "avg_batt_temp": {"strategy": "additive", "std": 0.2},
    "avg_amb_temp": {"strategy": "additive", "std": 0.75},
    "weekly_avg_batt_power": {"strategy": "relative", "std": 0.03},
    "weekly_distance": {"strategy": "relative", "std": 0.03, "clip": (0.0, None)},
    "weekly_regen_energy": {"strategy": "relative", "std": 0.05, "clip": (0.0, None)},
    "weekly_net_energy": {"strategy": "relative", "std": 0.04, "clip": (0.0, None)},
    "weekly_aux_energy": {"strategy": "relative", "std": 0.05, "clip": (0.0, None)},
    "weekly_avg_crate_chg": {"strategy": "additive", "std": 0.002, "clip": (0.0, None)},
    "weekly_cycles": {"strategy": "relative", "std": 0.04, "clip": (0.0, None)},
    "weekly_driving_time": {"strategy": "relative", "std": 0.04, "clip": (0.0, None)},
    "weekly_charging_time": {"strategy": "relative", "std": 0.04, "clip": (0.0, None)},
    "weekly_parking_time": {"strategy": "relative", "std": 0.04, "clip": (0.0, None)},
}

# Rebuild cumulative totals after noise so totals stay aligned with noised
# weekly values. Targets stay untouched by default.
CUMULATIVE_RECOMPUTE_MAP = {
    "weekly_distance": "weekly_tot_distance",
    "weekly_regen_energy": "weekly_tot_regen_energy",
    "weekly_net_energy": "weekly_tot_net_energy",
    "weekly_aux_energy": "weekly_tot_aux_energy",
    "weekly_cycles": "weekly_tot_cycles",
}
