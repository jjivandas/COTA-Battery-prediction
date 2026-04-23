# Implementation Summary: mar-2026-noisy-v1

## What was added

- A separate noisy dataset variant was created so the clean baseline stays untouched.
- Noise is injected into selected weekly input measurements during data processing.
- The same feature engineering, modeling, and evaluation pipeline was rerun on the noisy variant.
- A clean-vs-noisy comparison table was generated automatically.

## What stayed the same

- Same raw simulation source data
- Same engineered features
- Same two models in batch 1
- Same cumulative and delta targets
- Same GroupKFold, holdout, and temporal evaluations

## Headline result

- Best temporal result on the noisy dataset: `random_forest` / `delta` with temporal R2 `-0.6941`.
- Change from clean baseline for that run: `-0.2125` R2.

## How to explain this in slides

- We created a messier version of the same experiment to mimic measurement uncertainty.
- We did not change the models or scoring rules, so the comparison is fair.
- If performance only drops a little, the model is robust to noisy inputs.
- If performance drops a lot, the model depends too much on unrealistically clean simulation outputs.

## Files to open next

- `comparison_to_clean.csv` for the full metric-by-metric comparison
- `comparison_summary.md` for a short experiment-level readout
- `runs/` for per-model figures and predictions
