# Run: 004_random_forest_delta

**Date**: 2026-04-03
**Dataset**: mar-2026
**Model**: random_forest
**Target**: delta_qloss (delta)
**Samples**: 1739
**Features**: 27

## Results

| Split Strategy | R2 | RMSE | MAE | MAPE |
|---|---|---|---|---|
| GroupKFold CV | 0.9646 | 0.0120 | 0.0063 | 4.72% |
| Leave-Buses-Out | 0.9826 | 0.0084 | 0.0060 | 4.70% |
| Temporal 80/20 | -0.4816 | 0.0084 | 0.0063 | 8.22% |

## Features Used

- `avg_amb_temp`
- `avg_batt_temp`
- `avg_batt_temp_lag1`
- `avg_batt_temp_roll4`
- `avg_dod`
- `avg_soc`
- `charging_frac`
- `delta_qloss_lag1`
- `driving_frac`
- `max_soc`
- `min_soc`
- `parking_frac`
- `regen_ratio`
- `route_count`
- `soc_range`
- `temp_delta`
- `week_index`
- `weekly_aux_energy`
- `weekly_avg_batt_power`
- `weekly_avg_crate_chg`
- `weekly_charging_time`
- `weekly_distance`
- `weekly_distance_roll4`
- `weekly_driving_time`
- `weekly_net_energy`
- `weekly_parking_time`
- `weekly_regen_energy`

## Dropped Columns

- `week_start`
- `week_end`
- `routes`
- `bus_id`
- `is_service_week`
- `weekly_cycles`
- `avg_qloss_cycling`
- `avg_qloss_calendar`
- `delta_qloss_cycling`
- `delta_qloss_calendar`

## Files

- `config.json` — full reproducible configuration
- `metrics.json` — all metric scores
- `predictions.csv` — actual vs predicted for every test sample
- `figures/actual_vs_predicted.png` — scatter plot
- `figures/residuals.png` — residual analysis
- `figures/per_bus_trajectories.png` — per-bus prediction curves
- `figures/feature_importances.png` — top feature importances
