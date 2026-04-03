# Data Cleaning Pipeline

## Raw Data

- 12 per-bus CSVs located in `data/raw/<dataset>/per-bus-csvs/Bus-Id-001..012/`
- Each CSV: ~143-154 rows (one per week), 28 columns
- Column names are inconsistent — mixed case, units in brackets (e.g. `Max_SOC [-]`, `weeklyDistance [km]`)
- Date format: `01-Jan-2020` (day-month abbreviation-year)
- Bus-002 has a different filename (`sim_weekly_agg_002.csv` vs `sim_weekly_agg.csv`)

## Cleaning Steps (src/data_processing.py)

### 1. Column Renaming
All 28 columns renamed to consistent snake_case with units stripped:

```
Week_Index              → week_index
Max_SOC [-]             → max_soc
weeklyDistance [km]     → weekly_distance
Avg_Amb_Temp [degC]     → avg_amb_temp
weeklyDRegEn [kWh]      → weekly_regen_energy
weeklyparkingTime [h]   → weekly_parking_time
Avg_Qloss               → avg_qloss
...etc (full mapping in src/config.py COLUMN_RENAME)
```

### 2. Date Parsing
`week_start` and `week_end` converted from string `"01-Jan-2020"` to datetime objects using format `%d-%b-%Y`.

### 3. Bus ID
Integer `bus_id` (1-12) added to each row so bus identity is preserved after merging.

### 4. Service Week Flagging
`is_service_week = True` when `weekly_distance > 0`. Weeks where the bus was not in service get `False`.

Across the mar-2026 dataset: 1770 service weeks, 7 non-service weeks.

### 5. Forward-Filling Cumulative Columns
Cumulative columns are running totals that should never decrease within a bus. During no-service weeks, they show NaN in the raw data. We forward-fill within each bus group so the cumulative total carries through.

Columns forward-filled:
- `weekly_tot_distance`, `weekly_tot_regen_energy`, `weekly_tot_net_energy`
- `weekly_tot_aux_energy`, `weekly_tot_cycles`
- `avg_qloss_cycling`, `avg_qloss_calendar`, `avg_qloss`

Weekly columns (like `weekly_distance`) are NOT filled — NaN means "no data this week" and that's honest.

### 6. Sorting
Rows sorted by `(bus_id, week_index)` so each bus's time series is contiguous and chronologically ordered.

## Output

`data/processed/<dataset>/master_dataset.csv` — 1777 rows, 30 columns (28 original renamed + `bus_id` + `is_service_week`)
