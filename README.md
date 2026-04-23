# COTA Battery Degradation Prediction

ML pipeline to predict weekly battery capacity loss (`avg_qloss`) in 12 COTA electric buses using operational, environmental, and battery features.

## Project Structure

```
src/
├── config.py              # Paths, column mappings, constants. Change ACTIVE_DATASET here.
├── data_processing.py     # Load, clean, merge 12 per-bus CSVs → master_dataset.csv
├── feature_engineering.py # Compute 14 derived features → feature_matrix.csv
├── evaluation.py          # Metrics (RMSE, MAE, R², MAPE) + plotting utilities
└── modeling.py            # 9 models × 2 targets, GroupKFold CV + holdout evaluation

data/
├── raw/<dataset>/per-bus-csvs/Bus-Id-001..012/   # Raw simulation CSVs
└── processed/<dataset>/
    ├── master_dataset.csv    # Cleaned + merged (1777 rows, 30 cols)
    └── feature_matrix.csv   # With engineered features (1777 rows, 44 cols)

results/<dataset>/
├── model_comparison.csv
└── figures/
```

## Data Pipeline

```
Raw CSVs (12 buses) → data_processing.py (+ optional noise injection) → master_dataset.csv → feature_engineering.py → feature_matrix.csv
```

### Step 1: Data Processing (`python -m src.data_processing`)

- Reads 12 CSVs from `data/raw/<dataset>/per-bus-csvs/` (handles Bus-002's different filename)
- Renames all 28 columns to clean snake_case (strips units like `[km]`, `[kWh]`, `[-]`)
- Parses dates (`%d-%b-%Y`), adds integer `bus_id` (1–12)
- Flags no-service weeks (`is_service_week`) where distance = 0
- Forward-fills cumulative columns within each bus group
- Optionally injects column-specific measurement noise into weekly inputs, then recomputes cumulative totals
- Outputs: `data/processed/<dataset>/master_dataset.csv`

### Optional Noise Injection

Noise is configured in `src/config.py` and applied inside `src.data_processing` after cleaning and service-week flagging:

- Weekly measurements can be perturbed with additive or relative Gaussian noise
- Weekly time components are renormalized so driving + charging + parking stays constant for each week
- Cumulative totals are rebuilt from the noised weekly values
- Targets (`avg_qloss`, `avg_qloss_cycling`, `avg_qloss_calendar`) are left clean by default

The pipeline now supports parallel dataset variants through environment variables:

- `BASE_DATASET=mar-2026 DATASET_VARIANT=clean` → clean baseline
- `BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1` → noisy variant with separate processed/results outputs

When the noisy variant is active, the profile is also written to `data/processed/<dataset>/noise_config.json`.

### Step 2: Feature Engineering (`python -m src.feature_engineering`)

New features computed per bus:

| Feature | Description |
|---|---|
| `delta_qloss`, `delta_qloss_cycling`, `delta_qloss_calendar` | Week-over-week change in cumulative loss (`.diff()` per bus) |
| `soc_range` | `max_soc - min_soc` |
| `regen_ratio` | `weekly_regen_energy / weekly_net_energy` |
| `driving_frac`, `charging_frac`, `parking_frac` | Time utilization as fraction of total |
| `temp_delta` | `avg_batt_temp - avg_amb_temp` |
| `route_count` | Number of distinct routes per week |
| `delta_qloss_lag1`, `avg_batt_temp_lag1` | 1-week lagged values within each bus |
| `weekly_distance_roll4`, `avg_batt_temp_roll4` | 4-week rolling mean within each bus |

Outputs: `data/processed/<dataset>/feature_matrix.csv`

## Modeling (Step 3: `python -m src.modeling`)

**Two targets:**
- **Cumulative** (`avg_qloss`): total capacity loss to date — highly predictable but dominated by time trend
- **Delta** (`delta_qloss`): weekly incremental loss — lower R² but reveals which operational factors drive degradation

**9 models:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR, Bayesian Ridge, Gaussian Process

**3 evaluation strategies:**
1. GroupKFold (4 folds by `bus_id`) — primary model selection
2. Leave-buses-out (buses 10–12 held out) — generalization test
3. Temporal split (80/20 per bus) — time-ordering preserved

## Switching Datasets

When new data arrives:

1. Place it under `data/raw/<new-name>/per-bus-csvs/` (same Bus-Id-001..012 structure)
2. Run the pipeline with `BASE_DATASET=<new-name>`
3. Re-run the pipeline:
   ```bash
   BASE_DATASET=<new-name> DATASET_VARIANT=clean python -m src.data_processing
   BASE_DATASET=<new-name> DATASET_VARIANT=clean python -m src.feature_engineering
   BASE_DATASET=<new-name> DATASET_VARIANT=clean python -m src.modeling
   ```
   All outputs go to `data/processed/<new-name>/` and `results/<new-name>/` — previous results are untouched.

## Running Clean And Noisy Tracks

Clean baseline:

```bash
BASE_DATASET=mar-2026 DATASET_VARIANT=clean python -m src.data_processing
BASE_DATASET=mar-2026 DATASET_VARIANT=clean python -m src.feature_engineering
BASE_DATASET=mar-2026 DATASET_VARIANT=clean python -m src.modeling
```

Noisy variant:

```bash
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.data_processing
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.feature_engineering
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.modeling
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.comparison
```

## Current Dataset: mar-2026

- **12 buses**, 143–154 rows each, **1777 total rows**
- Date range: Jan 2020 – Oct 2023 (weekly aggregates from simulation)
- 7 no-service weeks across all buses
- Target `avg_qloss` range: 0.56 – 20.09
