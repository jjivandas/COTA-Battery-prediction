# Clean vs mar-2026-noisy-v1

This file compares the noisy experiment against the original clean baseline.

## What changed

- The model pipeline stayed the same.
- The evaluation splits stayed the same.
- The difference is that the noisy dataset perturbs selected input measurements before feature engineering.

## Metric deltas vs clean baseline

| Model | Target | CV R2 Δ | Holdout R2 Δ | Temporal R2 Δ | CV RMSE Δ | Holdout RMSE Δ | Temporal RMSE Δ |
|---|---|---|---|---|---|---|---|
| linear_regression | cumulative | -0.0006 | -0.0010 | -0.1461 | 0.0128 | 0.0223 | 0.0333 |
| linear_regression | delta | -0.0026 | -0.0014 | 0.6097 | 0.0004 | 0.0003 | -0.0009 |
| random_forest | cumulative | 0.0000 | -0.0000 | -0.0621 | -0.0021 | 0.0045 | 0.0103 |
| random_forest | delta | -0.0019 | 0.0003 | -0.2125 | 0.0003 | -0.0001 | 0.0006 |
