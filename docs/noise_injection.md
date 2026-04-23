# Noise Injection In Simple English

## What we are doing

We are making a second version of the same dataset where some input measurements are a little noisy.

This means:
- the original clean dataset stays as-is
- the noisy dataset is a separate copy for testing
- the rest of the pipeline stays the same

The point is to see whether the models still work when the inputs are a bit messy, like real-world sensor data.

## Where the noise is added

Noise is added during data processing.

That means:
1. we load the raw weekly simulation data
2. we clean the column names and parse dates
3. we mark service weeks
4. we fill the cumulative columns
5. then we add noise to selected input columns
6. after that, feature engineering and modeling run normally

Simple way to explain it:
- "We add the uncertainty early, so everything later uses the noisy data naturally."

## What kinds of columns get noise

We only add noise to input measurements that could realistically be a little off in practice.

These include:
- battery state values like SOC and DOD
- temperature readings
- weekly distance, energy, power, cycles, and time values

Examples:
- `avg_soc`
- `max_soc`
- `min_soc`
- `avg_dod`
- `avg_batt_temp`
- `avg_amb_temp`
- `weekly_distance`
- `weekly_regen_energy`
- `weekly_net_energy`
- `weekly_aux_energy`
- `weekly_avg_batt_power`
- `weekly_avg_crate_chg`
- `weekly_cycles`
- `weekly_driving_time`
- `weekly_charging_time`
- `weekly_parking_time`

## What kinds of columns do not get noise

We do not add noise to:
- IDs
- dates
- route labels
- service-week flags
- target values
- cumulative totals directly

Examples:
- `bus_id`
- `week_index`
- `week_start`
- `week_end`
- `routes`
- `is_service_week`
- `avg_qloss`
- `avg_qloss_cycling`
- `avg_qloss_calendar`

Simple way to explain it:
- "We are testing noisy inputs, not noisy answers."

## How the noise is added

We use two styles of noise.

### 1. Additive noise

Formula:
- `new value = old value + small random error`

Use this for values where the error is more like being a little high or a little low.

Good examples:
- temperature
- SOC
- DOD

Simple way to explain it:
- "The sensor reads a little above or below the real value."

### 2. Relative noise

Formula:
- `new value = old value * (1 + small random error)`

Use this for values where bigger readings should naturally allow bigger absolute error.

Good examples:
- weekly distance
- weekly energy
- weekly time

Simple way to explain it:
- "The reading is off by a small percentage."

## What happens to cumulative totals

We do not shake the cumulative totals directly.

Instead:
- we first noise the weekly values
- then we rebuild the totals from those weekly values

Examples:
- `weekly_tot_distance`
- `weekly_tot_regen_energy`
- `weekly_tot_net_energy`
- `weekly_tot_aux_energy`
- `weekly_tot_cycles`

Simple way to explain it:
- "We recalculate the running totals so they still match the noisy weekly data."

## How we keep the noisy data realistic

After adding noise, we repair anything that becomes physically inconsistent.

That includes:
- keeping `min_soc <= avg_soc <= max_soc`
- keeping nonnegative quantities nonnegative
- keeping driving + charging + parking time equal to the original weekly total
- keeping cumulative totals increasing properly within each bus

Simple way to explain it:
- "We want the data to be messy, but still believable."

## Why we keep clean and noisy results separate

We save the noisy pipeline outputs in a separate dataset/result folder.

That way:
- the clean baseline stays untouched
- the comparison is fair
- we can say exactly how much performance changed because of noisy inputs

Simple way to explain it:
- "We are making a side-by-side experiment, not replacing the original one."
