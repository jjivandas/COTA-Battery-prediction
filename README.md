# COTA Battery Degradation Prediction

A research pipeline built with the Center for Automotive Research (CAR) at The Ohio State University to predict weekly battery capacity loss (`avg_qloss`) on twelve Central Ohio Transit Authority (COTA) electric buses.

> **Important context**: the data we model is **simulated** by the CAR team — it is *not* live operational telemetry from the COTA fleet. The goal of this codebase is to build modeling and evaluation infrastructure on the simulated data so that, when real-world data becomes available, the same pipeline can be applied with minimal change.

This README is written as a **handoff document**. It explains every directory, every file, every config knob, and how to reproduce every result. The companion report in `docs/REPORT.md` covers the *why* of the project, the experiments we ran, and what we found.

---

## Table of contents

1. [Project context](#1-project-context)
2. [Repository layout](#2-repository-layout)
3. [Environment setup](#3-environment-setup)
4. [The dataset](#4-the-dataset)
5. [Configuration system](#5-configuration-system)
6. [Pipeline stage 1 — data processing](#6-pipeline-stage-1--data-processing)
7. [Optional noise injection](#7-optional-noise-injection)
8. [Pipeline stage 2 — feature engineering](#8-pipeline-stage-2--feature-engineering)
9. [Pipeline stage 3 — modeling](#9-pipeline-stage-3--modeling)
10. [Evaluation strategies](#10-evaluation-strategies)
11. [Run-based reproducibility](#11-run-based-reproducibility)
12. [Comparison: clean vs noisy](#12-comparison-clean-vs-noisy)
13. [Standalone scripts](#13-standalone-scripts)
14. [Reproducing every result from scratch](#14-reproducing-every-result-from-scratch)
15. [Extending the pipeline](#15-extending-the-pipeline)
16. [Documentation index](#16-documentation-index)

---

## 1. Project context

- **Collaboration**: CAR ↔ COTA. CAR generates simulated weekly aggregates of bus operation; this repo turns those simulations into supervised-learning experiments.
- **Prediction target**: `Avg_Qloss` — the cumulative capacity-loss percentage (a proxy for State of Health) for each bus at the end of each week.
- **Why simulated data first**: real telemetry is expensive, sparse, and lagging. Simulation lets us prototype the modeling stack, define evaluation rules, and understand which input features actually drive predicted degradation before we commit to instrumentation.
- **End-state vision**: feed the same pipeline live on-vehicle data and produce a continuously updated degradation forecast for each bus.

## 2. Repository layout

```
COTA-Battery-prediction/
├── README.md                 # this file
├── LICENSE
├── requirements.txt          # Python dependencies (pinned for reproducibility)
├── venv/                     # local virtualenv (created by you, gitignored)
│
├── data/
│   ├── raw/<base>/           # raw simulation CSVs from CAR
│   │   ├── docs/             # CAR data-description PDFs
│   │   └── per-bus-csvs/Bus-Id-001..012/sim_weekly_agg.csv
│   └── processed/<active>/   # cleaned + engineered outputs (per dataset variant)
│       ├── master_dataset.csv
│       ├── feature_matrix.csv
│       ├── dataset_config.json
│       └── noise_config.json   (only for noisy variants)
│
├── src/                      # ALL pipeline code lives here
│   ├── config.py             # paths, column maps, constants, noise profile
│   ├── data_processing.py    # raw CSVs -> master_dataset.csv
│   ├── noise.py              # noise injection (used only when variant != clean)
│   ├── feature_engineering.py  # master_dataset.csv -> feature_matrix.csv
│   ├── modeling.py           # train/evaluate/save runs
│   ├── evaluation.py         # metrics + plotting helpers
│   └── comparison.py         # clean-vs-noisy delta tables and summaries
│
├── scripts/
│   └── make_forecast_plot.py # train/forecast visualization (per bus)
│
├── results/<active>/         # all model outputs for a given dataset variant
│   ├── runs/
│   │   ├── 001_linear_regression_cumulative/
│   │   │   ├── config.json         # full reproducible config snapshot
│   │   │   ├── metrics.json        # all CV / holdout / temporal scores
│   │   │   ├── predictions.csv     # actual + predicted + residual per row
│   │   │   ├── summary.md          # human-readable run summary
│   │   │   └── figures/            # actual_vs_predicted, residuals, trajectories, etc.
│   │   ├── 002_..._delta/
│   │   ├── 003_random_forest_cumulative/
│   │   ├── 004_random_forest_delta/
│   │   └── run_log.csv             # one row per run (for cross-run analysis)
│   ├── figures/                # standalone figures from scripts/
│   └── (noisy variants only) comparison_to_clean.csv,
│                              comparison_summary.md,
│                              implementation_summary.md
│
├── docs/                     # design docs and the project report
│   ├── REPORT.md             # the deep-detail project report
│   ├── data_cleaning.md
│   ├── feature_engineering.md
│   ├── models.md
│   ├── model_explainer.md
│   ├── evaluation_strategy.md
│   ├── noise_injection.md
│   └── debugging_log.md
│
└── notebooks/                # exploratory notebooks (optional, not part of the pipeline)
```

The pipeline is **intentionally script-based, not notebook-based**: every result must be reproducible from the command line so that reviewers and future maintainers can re-run anything without state-leak from a notebook kernel.

## 3. Environment setup

Python 3.11+ recommended. The project uses a local virtualenv to insulate it from the system Python (we hit NumPy 1.x/2.x ABI mismatches against the system anaconda).

```bash
cd COTA-Battery-prediction
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

All pipeline commands assume the venv is active and that the working directory is the repo root. Modules are invoked with `python -m src.<name>` so that relative imports resolve.

## 4. The dataset

### 4.1 Source

Twelve simulated buses (Bus-Id-001 .. Bus-Id-012). Each bus has one CSV containing weekly aggregates over Jan 2020 – Oct 2023. Bus-002 has a different filename (`sim_weekly_agg_002.csv`) — handled in `config.get_csv_path`.

Total rows after concatenation: **1777**. Each bus contributes 143–154 weekly rows.

### 4.2 Raw column dictionary (per CAR's `00-Description of Data.pdf`)

The renamer in `src/config.py:COLUMN_RENAME` strips units and converts to snake_case. Renamed columns:

| Renamed column | Raw name | Meaning |
|---|---|---|
| `week_index` | Week_Index | Week counter, **independent per bus** |
| `week_start`, `week_end` | Week_Start / Week_End | ISO-style date range for the row |
| `routes` | Routes | Comma-separated route IDs operated that week |
| `avg_soc`, `max_soc`, `min_soc` | Avg/Max/Min_SOC [-] | State-of-charge stats over the week |
| `avg_dod` | Avg_DOD [-] | Average depth of discharge |
| `avg_batt_temp` | Avg_Batt_Temp | Average battery cell temperature (°C) |
| `avg_amb_temp` | Avg_Amb_Temp [degC] | Average ambient temperature |
| `weekly_avg_batt_power` | weeklyAvgBattP | Average battery power |
| `weekly_distance` / `weekly_tot_distance` | weeklyDistance / weeklyTotDistance [km] | Distance for the week and the cumulative total |
| `weekly_regen_energy` / `weekly_tot_regen_energy` | weeklyDRegEn / weeklyTotRegEn [kWh] | Regenerative-braking energy, weekly and cumulative |
| `weekly_net_energy` / `weekly_tot_net_energy` | weeklyDNetEn / weeklyTotNetEn [kWh] | Net energy, weekly and cumulative |
| `weekly_aux_energy` / `weekly_tot_aux_energy` | weeklyEnAux / weeklyTotAux [kWh] | Auxiliary load energy |
| `weekly_avg_crate_chg` | weeklyAvgCrateChg [1/h] | Average charging C-rate |
| `weekly_cycles` / `weekly_tot_cycles` | weeklyCycles / weeklyTotCycles [-] | Equivalent full cycles, weekly and cumulative |
| `weekly_driving_time`, `weekly_charging_time`, `weekly_parking_time` | weeklyDrivingTime / ChargingTime / parkingTime [h] | Time budget for the week (hours) |
| `avg_qloss_cycling` | Avg_Qcyc | Cycling-aging contribution to capacity loss |
| `avg_qloss_calendar` | Avg_Qcal | Calendar-aging contribution to capacity loss |
| `avg_qloss` | Avg_Qloss | **Target**: total capacity loss (cycling + calendar) |

Two important properties of the raw target:

- `avg_qloss = avg_qloss_cycling + avg_qloss_calendar` (decomposition into use-driven and time-driven aging).
- All three are **monotonically increasing within a bus** — they are cumulative capacity loss to date, not a per-week increment.

## 5. Configuration system

Everything you might want to change without editing pipeline code lives in `src/config.py`.

### 5.1 Dataset selection (env vars)

The pipeline supports parallel dataset *variants* so we can run a clean baseline and one or more noisy experiments side-by-side without overwriting outputs.

```python
BASE_DATASET    = os.getenv("BASE_DATASET",    "mar-2026")  # which raw drop
DATASET_VARIANT = os.getenv("DATASET_VARIANT", "clean")     # which output namespace
ACTIVE_DATASET  = BASE_DATASET if variant == "clean" else f"{BASE_DATASET}-{variant}"
```

| Setting | Reads from | Writes to |
|---|---|---|
| `BASE_DATASET=mar-2026 DATASET_VARIANT=clean` | `data/raw/mar-2026/` | `data/processed/mar-2026/`, `results/mar-2026/` |
| `BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1` | `data/raw/mar-2026/` | `data/processed/mar-2026-noisy-v1/`, `results/mar-2026-noisy-v1/` |

Switching datasets does **not** require code changes — only env vars.

### 5.2 Other config knobs

- `COLUMN_RENAME` — raw → snake_case mapping (see §4.2).
- `CUMULATIVE_COLS` / `WEEKLY_COLS` — column-group lists used across the pipeline (e.g. forward-fill scope, leakage exclusion).
- `TARGET_CUMULATIVE = "avg_qloss"`, `TARGET_DELTA = "delta_qloss"` — the two prediction targets we model.
- `NOISE_*` and `COLUMN_NOISE` — the noise profile (see §7).
- `CUMULATIVE_RECOMPUTE_MAP` — pairs of (weekly column → cumulative column) so totals can be rebuilt after weekly values are perturbed.

## 6. Pipeline stage 1 — data processing

**Entry point**: `python -m src.data_processing` &nbsp; (lives in `src/data_processing.py`)

What it does, in order:

1. **Load each bus** (`load_single_bus`) — read its CSV, rename columns via `COLUMN_RENAME`, parse dates with format `%d-%b-%Y`, attach `bus_id` as an integer.
2. **Concatenate all 12 buses** (`load_all_buses`) into a single dataframe sorted by `(bus_id, week_index)`.
3. **Flag service weeks** (`flag_service_weeks`): `is_service_week = weekly_distance > 0`. We later filter on this so weeks where the bus didn't operate don't pollute the training set with zero-input rows.
4. **Forward-fill cumulative columns** (`handle_missing_values`) within each bus group only — preserves monotonicity of cumulative totals across no-service weeks. Targets are intentionally left raw.
5. **Optionally apply noise** (when `DATASET_VARIANT != "clean"`) via `src.noise.apply_noise_profile` — see §7.
6. **Write outputs** to `data/processed/<active>/`:
   - `master_dataset.csv` — cleaned, merged, optionally noised
   - `dataset_config.json` — records `base_dataset`, `active_dataset`, `noise_enabled`
   - `noise_config.json` — full noise profile snapshot (noisy variants only)
7. **Print a quality summary**: row counts per bus, date range, missing values, service-week counts, target range.

## 7. Optional noise injection

**Entry point**: `src/noise.py:apply_noise_profile` (called automatically from data processing when the variant ≠ `clean`).

The clean dataset is "too perfect" — simulated weekly aggregates have no measurement uncertainty. The noise module simulates realistic sensor uncertainty so we can measure how robust each model is to dirtier inputs.

### 7.1 What gets perturbed

`COLUMN_NOISE` in `src/config.py` defines per-column rules. Each entry has:

- `strategy`: `"additive"` (`x + N(0, std)`) or `"relative"` (`x * (1 + N(0, std))`).
- `std`: standard deviation in units of the column or as a fraction.
- `clip` (optional): hard bounds, e.g. `(0.0, 1.0)` for SOC or `(0.0, None)` for non-negative quantities.

Highlights of the default profile (`default_sensor_noise_v1`, seed 42):

- SOC stats get small additive noise (e.g. `avg_soc` gets σ=0.003).
- Temperatures get additive noise (battery σ=0.2 °C, ambient σ=0.75 °C).
- Energy / distance / cycles / time use relative noise (3–5%).

### 7.2 Consistency repairs

After raw perturbation we run three repairs so the noisy dataset is still self-consistent:

1. **`_repair_soc_columns`** — re-sorts `(min_soc, avg_soc, max_soc)` row-wise so the ordering invariant holds.
2. **`_normalize_weekly_time_budget`** — driving + charging + parking hours are rescaled per row to match the *original* total (i.e. noise redistributes how the week was spent, but doesn't change how long the week was).
3. **`_recompute_cumulative_columns`** — totals like `weekly_tot_distance` are re-derived as `cumsum` of the noised weekly values, so cumulative columns can't disagree with their weekly counterparts.

### 7.3 What is *not* perturbed

The targets — `avg_qloss`, `avg_qloss_cycling`, `avg_qloss_calendar` — are left clean. We're testing model robustness to noisy *inputs*, not relabeling the truth.

Service-week-only mode is on by default (`NOISE_SERVICE_WEEKS_ONLY = True`): we skip noising idle weeks where everything is zero.

## 8. Pipeline stage 2 — feature engineering

**Entry point**: `python -m src.feature_engineering`

Reads `master_dataset.csv`, adds the following features, writes `feature_matrix.csv`:

| Feature | Formula |
|---|---|
| `delta_qloss` | per-bus `avg_qloss.diff()` (week-over-week increment) |
| `delta_qloss_cycling` | per-bus `avg_qloss_cycling.diff()` |
| `delta_qloss_calendar` | per-bus `avg_qloss_calendar.diff()` |
| `soc_range` | `max_soc - min_soc` |
| `regen_ratio` | `weekly_regen_energy / weekly_net_energy` (zero when net=0) |
| `driving_frac`, `charging_frac`, `parking_frac` | each time component / total weekly time |
| `temp_delta` | `avg_batt_temp - avg_amb_temp` |
| `route_count` | number of comma-separated entries in `routes` |
| `delta_qloss_lag1` | per-bus 1-week lag of `delta_qloss` |
| `avg_batt_temp_lag1` | per-bus 1-week lag of `avg_batt_temp` |
| `weekly_distance_roll4` | per-bus 4-week rolling mean of `weekly_distance` |
| `avg_batt_temp_roll4` | per-bus 4-week rolling mean of `avg_batt_temp` |

The `delta_*` columns are computed here (not in data processing) so that when noise is applied to weekly values, deltas reflect the noised cumulative target.

## 9. Pipeline stage 3 — modeling

**Entry point**: `python -m src.modeling` (calls `run_batch_1`)

### 9.1 Models

Defined in `get_model(name)`:

- **`linear_regression`** — `Pipeline([StandardScaler, LinearRegression])`. Scaling matters here because coefficients are interpreted on standardized features in the diagnostic plot.
- **`random_forest`** — `Pipeline([RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)])`. No scaler — trees are scale-invariant.

Both wrapped in a `Pipeline` so that `clone(pipeline)` gives a fresh untrained estimator per fold.

### 9.2 Targets

Two distinct prediction problems:

- **`cumulative`** (`avg_qloss`): predict total capacity loss to date. High R² achievable but dominated by the time trend — this is "where on the curve are we?".
- **`delta`** (`delta_qloss`): predict the weekly *increment*. Harder, but the only target where the input features can really shine — we want the model to say "this week's driving conditions cause this much extra aging."

`get_feature_cols(df, target_type)` excludes everything in `DROP_COLS` plus, for delta targets, every cumulative column (otherwise the model just learns `avg_qloss(t-1) - avg_qloss(t-1)` ≈ delta and you get illusory near-perfect R²).

### 9.3 The feature gating logic (leakage prevention)

`DROP_COLS` in `modeling.py`:

```
week_start, week_end                       # raw dates; we keep week_index instead
routes                                     # string; encoded as route_count
bus_id                                     # group identifier, not a predictor
is_service_week                            # used to filter, not predict
weekly_cycles                              # 97.5% are 7.0, near-constant noise
avg_qloss_cycling, avg_qloss_calendar      # sub-components of avg_qloss target
delta_qloss_cycling, delta_qloss_calendar  # sub-components of delta_qloss target
```

The qloss sub-components are dropped as inputs because `avg_qloss = cycling + calendar` — including either as a feature would let the model trivially reconstruct the target.

### 9.4 Data preparation

`prepare_data(df, feature_cols, target_col)`:

1. Filter to service weeks (`is_service_week = True`).
2. Build the working column set: `feature_cols + [target_col] + meta_cols`, where `meta_cols` is `["bus_id", "week_index"]` minus anything already in `feature_cols` (this dedupe step prevents a 3-D NumPy array bug we hit early on).
3. `dropna()` — for the delta target, this removes the first week of each bus (where `diff()` is NaN) and any rows touched by the lag/rolling features that haven't accumulated history.
4. Return `(X, y, bus_ids, week_indices, valid_idx)`. The `.ravel()` on `bus_ids` and `week_indices` guarantees they're 1-D regardless of duplicate-column quirks upstream.

## 10. Evaluation strategies

Every run is evaluated under three different splits, because each split answers a different question.

| Split | Function | Question it answers |
|---|---|---|
| `GroupKFold` (4 folds by `bus_id`) | `split_group_kfold` | "Does the model generalize from one bus to another?" |
| Leave-buses-out (10, 11, 12 held out) | `split_leave_buses_out` | "Hard generalization: trained on buses 1–9, scored on three buses it has never seen." |
| Temporal 80/20 per bus | `split_temporal` | "Can the model forecast forward in time? Trained on the first 80% of each bus's weeks, scored on the last 20%." |

All three are run in `execute_run`. Metrics computed: `RMSE`, `MAE`, `R²`, `MAPE` — see `src/evaluation.py:compute_metrics`.

> **Negative R² is meaningful**: it means the model performs *worse than predicting the mean of the test set*. We see this consistently on the temporal split — see the report.

## 11. Run-based reproducibility

Each call to `execute_run(model_name, target_type)` does the following:

1. Picks the next run number (`_next_run_number`) based on existing folders in `results/<active>/runs/`.
2. Creates `results/<active>/runs/NNN_<model>_<target>/`.
3. Saves a complete `config.json` snapshot: model, target, dataset, variant, noise flag, feature list, dropped columns, holdout buses, splits, seeds, sklearn params, timestamp.
4. Runs all three evaluation splits, collects metrics into `metrics.json`.
5. Saves `predictions.csv` (CV predictions for every row + temporal-test predictions).
6. Generates figures into `figures/`:
   - `actual_vs_predicted.png` — CV-fold scatter
   - `residuals.png` — CV-fold residuals
   - `per_bus_trajectories.png` — actual vs predicted curves per bus
   - For LR: `coefficients.png`. For RF: `feature_importances.png`.
7. Writes `summary.md` — a per-run human-readable summary.
8. Appends a row to `runs/run_log.csv` so all runs are queryable in one place.

`run_batch_1()` runs the four current baseline experiments: LR-cumulative, LR-delta, RF-cumulative, RF-delta.

## 12. Comparison: clean vs noisy

**Entry point**: `python -m src.comparison`  (only valid for `DATASET_VARIANT != "clean"`)

`build_comparison()` joins `results/<base>/runs/run_log.csv` (clean) against `results/<active>/runs/run_log.csv` (current, noisy) on `(model, target)` and computes a `_delta_vs_clean` for every metric across all three splits.

Output files (written to `results/<active>/`):

- `comparison_to_clean.csv` — full metric-by-metric delta table.
- `comparison_summary.md` — short markdown table of R² and RMSE deltas.
- `implementation_summary.md` — a slide-friendly "what we did vs the clean baseline" writeup with a headline result.

## 13. Standalone scripts

### `scripts/make_forecast_plot.py`

For a chosen model on the cumulative target, trains on the first 80% of each bus's weeks and forecasts the last 20%. Plots, per bus:

- the actual cumulative qloss trajectory across all weeks (blue),
- the model's fit on the train portion (green dashed),
- the model's *forecast* on the held-out portion (red dashed),
- a vertical boundary line marking the train/forecast split.

This is the visualization that makes the negative temporal R² intuitive: even when the train fit is excellent, the forecast diverges immediately. Output goes to `results/<active>/figures/forecast_<model>_cumulative.png`.

Run with:
```bash
python scripts/make_forecast_plot.py
```
(Generates plots for both `linear_regression` and `random_forest`.)

## 14. Reproducing every result from scratch

```bash
# 0. environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. CLEAN baseline (mar-2026)
BASE_DATASET=mar-2026 DATASET_VARIANT=clean python -m src.data_processing
BASE_DATASET=mar-2026 DATASET_VARIANT=clean python -m src.feature_engineering
BASE_DATASET=mar-2026 DATASET_VARIANT=clean python -m src.modeling

# 2. NOISY variant (mar-2026-noisy-v1)
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.data_processing
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.feature_engineering
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.modeling
BASE_DATASET=mar-2026 DATASET_VARIANT=noisy-v1 python -m src.comparison

# 3. forecast-vs-actual visualization (active variant)
python scripts/make_forecast_plot.py
```

When a new raw drop arrives:

1. Place it under `data/raw/<new-name>/per-bus-csvs/Bus-Id-001..012/`.
2. Re-run the three pipeline modules above with `BASE_DATASET=<new-name>`.

Old datasets remain untouched in their own `data/processed/<old>/` and `results/<old>/` folders.

## 15. Extending the pipeline

- **Add a new model**: register it in `modeling.get_model` and add a `(model_name, target_type)` tuple to `run_batch_1`. Make sure it's a `Pipeline` so `clone()` works in CV.
- **Add a new feature**: add a `compute_*` function in `feature_engineering.py`, call it from `build_feature_matrix`, and decide whether it belongs in `cumulative_features`, `delta_features`, or both via `get_feature_sets`.
- **Add a new noise variant**: copy `COLUMN_NOISE` to a new profile name and switch on `NOISE_PROFILE_NAME` in `config.py` (or pass via env var). Use a new `DATASET_VARIANT` so outputs land in their own folder.
- **Add a new evaluation split**: add a `split_*` function in `modeling.py` and a third block in `execute_run` that prefixes its metrics with a unique tag.
- **Vary simulation parameters** (next major experiment, see report): add a new raw dataset under `data/raw/<param-sweep-name>/` and run the pipeline with that base. The infrastructure does not change.

## 16. Documentation index

In `docs/`:

- **`REPORT.md`** — the deep-detail project report (what / how / why / results / next steps).
- `data_cleaning.md` — design notes for the data-processing stage.
- `feature_engineering.md` — rationale for each engineered feature.
- `models.md` — model selection notes.
- `model_explainer.md` — plain-English overview of LR vs RF.
- `evaluation_strategy.md` — why we use three splits and what each tells us.
- `noise_injection.md` — the noise profile, design choices, and consistency rules.
- `debugging_log.md` — bugs we hit and how we fixed them (3-D array, NumPy ABI, bus_id type, etc.).

## License

See `LICENSE`.
