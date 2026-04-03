# Run: 001_linear_regression_cumulative

**Date**: 2026-04-03
**Dataset**: mar-2026
**Model**: linear_regression
**Target**: avg_qloss (cumulative)
**Samples**: 1739
**Features**: 32

## Results

| Split Strategy | R2 | RMSE | MAE | MAPE |
|---|---|---|---|---|
| GroupKFold CV | 0.9888 | 0.5223 | 0.4151 | 5.10% |
| Leave-Buses-Out | 0.9874 | 0.5506 | 0.4766 | 5.08% |
| Temporal 80/20 | -1.3391 | 1.0828 | 1.0235 | 5.37% |

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
- `weekly_tot_aux_energy`
- `weekly_tot_cycles`
- `weekly_tot_distance`
- `weekly_tot_net_energy`
- `weekly_tot_regen_energy`

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
- `figures/coefficients.png` — feature coefficient magnitudes
