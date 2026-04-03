# Feature Engineering

## Input
`data/processed/<dataset>/master_dataset.csv` (1777 rows, 30 columns)

## New Features (src/feature_engineering.py)

### Delta Targets — The Core Insight

`avg_qloss` is cumulative (monotonically increasing over time). A bus at week 100 will always have more total loss than at week 10. Predicting it directly is easy but not very useful — it's mostly just a function of age.

The **delta** (week-over-week difference) tells us how much NEW degradation happened this specific week. That's what matters for understanding which operational factors drive degradation.

| Feature | Formula | Why |
|---|---|---|
| `delta_qloss` | `avg_qloss.diff()` per bus | Weekly incremental capacity loss — primary delta target |
| `delta_qloss_cycling` | `avg_qloss_cycling.diff()` per bus | Cycling-induced portion of weekly loss |
| `delta_qloss_calendar` | `avg_qloss_calendar.diff()` per bus | Calendar aging portion of weekly loss |

### Derived Operational Features

| Feature | Formula | Physical meaning |
|---|---|---|
| `soc_range` | `max_soc - min_soc` | How wide the state-of-charge swing is — wider swings stress the battery more |
| `regen_ratio` | `weekly_regen_energy / weekly_net_energy` | Fraction of energy recovered via regenerative braking — indicates route hilliness and driving style |
| `driving_frac` | `weekly_driving_time / total_time` | Fraction of week spent driving |
| `charging_frac` | `weekly_charging_time / total_time` | Fraction of week spent charging |
| `parking_frac` | `weekly_parking_time / total_time` | Fraction of week parked (idle calendar aging) |
| `temp_delta` | `avg_batt_temp - avg_amb_temp` | How much hotter the battery runs vs environment — indicates thermal stress from usage |
| `route_count` | Count of comma-separated routes | Operational complexity/variety per week |

### Temporal Features — Capturing Trends

| Feature | Formula | Why |
|---|---|---|
| `delta_qloss_lag1` | Previous week's `delta_qloss` (per bus) | Recent degradation rate predicts near-future rate |
| `avg_batt_temp_lag1` | Previous week's `avg_batt_temp` (per bus) | Thermal history affects current degradation |
| `weekly_distance_roll4` | 4-week rolling mean of `weekly_distance` (per bus) | Smooths out week-to-week noise in usage, captures sustained load |
| `avg_batt_temp_roll4` | 4-week rolling mean of `avg_batt_temp` (per bus) | Sustained thermal exposure matters more than single-week spikes |

All lags and rolling windows look **backward only** (no future data leakage). Rolling uses `min_periods=1` so early weeks still get values.

## Feature Sets for Modeling

Two separate feature sets are defined to prevent leakage:

### Cumulative model features (predicting `avg_qloss`)
- `week_index` (age)
- Cumulative columns: `weekly_tot_distance`, `weekly_tot_regen_energy`, `weekly_tot_net_energy`, `weekly_tot_aux_energy`, `weekly_tot_cycles`
- All weekly columns + all engineered features

### Delta model features (predicting `delta_qloss`)
- `week_index` (age proxy only)
- Weekly columns only — NO cumulative columns (they'd leak target information)
- All engineered features

## Output

`data/processed/<dataset>/feature_matrix.csv` — 1777 rows, 44 columns (30 original + 14 engineered)
