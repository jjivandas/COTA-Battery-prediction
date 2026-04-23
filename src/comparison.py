"""Compare noisy run outputs against the clean baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ACTIVE_DATASET, BASE_DATASET, DATASET_VARIANT, PROJECT_ROOT, RESULTS_DIR

SPLIT_PREFIXES = ["cv", "holdout", "temporal"]
METRICS = ["RMSE", "MAE", "R2", "MAPE"]


def build_comparison() -> tuple[pd.DataFrame, Path]:
    """Compare the active dataset's run log to the clean baseline."""
    clean_log = PROJECT_ROOT / "results" / BASE_DATASET / "runs" / "run_log.csv"
    current_log = RESULTS_DIR / "runs" / "run_log.csv"

    if not clean_log.exists():
        raise FileNotFoundError(f"Clean baseline run log not found: {clean_log}")
    if not current_log.exists():
        raise FileNotFoundError(f"Current run log not found: {current_log}")

    clean = pd.read_csv(clean_log)
    current = pd.read_csv(current_log)

    merge_cols = ["model", "target"]
    comparison = current.merge(clean, on=merge_cols, suffixes=("_current", "_clean"))
    comparison.insert(0, "dataset", ACTIVE_DATASET)
    comparison.insert(1, "dataset_variant", DATASET_VARIANT)

    for prefix in SPLIT_PREFIXES:
        for metric in METRICS:
            current_col = f"{prefix}_{metric}_current"
            clean_col = f"{prefix}_{metric}_clean"
            comparison[f"{prefix}_{metric}_delta_vs_clean"] = (
                comparison[current_col] - comparison[clean_col]
            )

    out_path = RESULTS_DIR / "comparison_to_clean.csv"
    comparison.to_csv(out_path, index=False)
    return comparison, out_path


def write_comparison_summary(comparison: pd.DataFrame) -> Path:
    """Write a plain-English comparison summary markdown file."""
    out_path = RESULTS_DIR / "comparison_summary.md"
    lines = [
        f"# Clean vs {ACTIVE_DATASET}",
        "",
        "This file compares the noisy experiment against the original clean baseline.",
        "",
        "## What changed",
        "",
        "- The model pipeline stayed the same.",
        "- The evaluation splits stayed the same.",
        "- The difference is that the noisy dataset perturbs selected input measurements before feature engineering.",
        "",
        "## Metric deltas vs clean baseline",
        "",
        "| Model | Target | CV R2 Δ | Holdout R2 Δ | Temporal R2 Δ | CV RMSE Δ | Holdout RMSE Δ | Temporal RMSE Δ |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, row in comparison.iterrows():
        lines.append(
            "| "
            f"{row['model']} | {row['target']} | "
            f"{row['cv_R2_delta_vs_clean']:.4f} | "
            f"{row['holdout_R2_delta_vs_clean']:.4f} | "
            f"{row['temporal_R2_delta_vs_clean']:.4f} | "
            f"{row['cv_RMSE_delta_vs_clean']:.4f} | "
            f"{row['holdout_RMSE_delta_vs_clean']:.4f} | "
            f"{row['temporal_RMSE_delta_vs_clean']:.4f} |"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def write_implementation_summary(comparison: pd.DataFrame) -> Path:
    """Write a short slide-friendly summary of the noisy-track work."""
    out_path = RESULTS_DIR / "implementation_summary.md"

    temporal = comparison[["model", "target", "temporal_R2_clean", "temporal_R2_current", "temporal_R2_delta_vs_clean"]]
    best_temporal = temporal.sort_values("temporal_R2_current", ascending=False).iloc[0]

    lines = [
        f"# Implementation Summary: {ACTIVE_DATASET}",
        "",
        "## What was added",
        "",
        "- A separate noisy dataset variant was created so the clean baseline stays untouched.",
        "- Noise is injected into selected weekly input measurements during data processing.",
        "- The same feature engineering, modeling, and evaluation pipeline was rerun on the noisy variant.",
        "- A clean-vs-noisy comparison table was generated automatically.",
        "",
        "## What stayed the same",
        "",
        "- Same raw simulation source data",
        "- Same engineered features",
        "- Same two models in batch 1",
        "- Same cumulative and delta targets",
        "- Same GroupKFold, holdout, and temporal evaluations",
        "",
        "## Headline result",
        "",
        f"- Best temporal result on the noisy dataset: `{best_temporal['model']}` / `{best_temporal['target']}` with temporal R2 `{best_temporal['temporal_R2_current']:.4f}`.",
        f"- Change from clean baseline for that run: `{best_temporal['temporal_R2_delta_vs_clean']:+.4f}` R2.",
        "",
        "## How to explain this in slides",
        "",
        "- We created a messier version of the same experiment to mimic measurement uncertainty.",
        "- We did not change the models or scoring rules, so the comparison is fair.",
        "- If performance only drops a little, the model is robust to noisy inputs.",
        "- If performance drops a lot, the model depends too much on unrealistically clean simulation outputs.",
        "",
        "## Files to open next",
        "",
        "- `comparison_to_clean.csv` for the full metric-by-metric comparison",
        "- `comparison_summary.md` for a short experiment-level readout",
        "- `runs/` for per-model figures and predictions",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    """Build all comparison artifacts for the active dataset."""
    if DATASET_VARIANT == "clean":
        raise ValueError("Comparison is only meaningful for a non-clean dataset variant.")

    comparison, csv_path = build_comparison()
    summary_path = write_comparison_summary(comparison)
    impl_path = write_implementation_summary(comparison)

    print(f"Saved comparison table to {csv_path}")
    print(f"Saved comparison summary to {summary_path}")
    print(f"Saved implementation summary to {impl_path}")


if __name__ == "__main__":
    main()
