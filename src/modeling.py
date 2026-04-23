"""Run-based model training with full reproducibility and visualization."""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from src.config import (
    ACTIVE_DATASET, BASE_DATASET, DATASET_VARIANT, FEATURE_MATRIX_PATH,
    NOISE_ENABLED, RESULTS_DIR,
    TARGET_CUMULATIVE, TARGET_DELTA, CUMULATIVE_COLS, WEEKLY_COLS,
)
from src.evaluation import (
    compute_metrics, plot_actual_vs_predicted, plot_residuals,
    plot_prediction_trajectories, plot_feature_importances,
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
RUNS_DIR = RESULTS_DIR / "runs"

# ── Columns excluded from modeling ─────────────────────────────────────────
DROP_COLS = [
    "week_start", "week_end",       # dates — temporal info in week_index
    "routes",                        # string — already encoded as route_count
    "bus_id",                        # group identifier, not a predictor
    "is_service_week",               # used for filtering, not prediction
    "weekly_cycles",                 # near-constant (97.5% are 7.0)
    "avg_qloss_cycling",             # sub-component of avg_qloss target
    "avg_qloss_calendar",            # sub-component of avg_qloss target
    "delta_qloss_cycling",           # sub-component of delta_qloss target
    "delta_qloss_calendar",          # sub-component of delta_qloss target
]


# ── Available models ───────────────────────────────────────────────────────

def get_model(name: str):
    """Return (pipeline, param_dict) for a named model."""
    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=20,
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name]


# ── Feature set selection ──────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, target_type: str) -> list:
    """Return feature columns based on target type, excluding drop cols."""
    all_cols = set(df.columns) - set(DROP_COLS) - {TARGET_CUMULATIVE, TARGET_DELTA}

    if target_type == "delta":
        # No cumulative columns — avoid leakage
        cum_set = set(CUMULATIVE_COLS)
        all_cols = all_cols - cum_set

    feature_cols = sorted(all_cols)
    return feature_cols


# ── Data preparation ───────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Filter service weeks, drop NaN rows, return X, y, metadata."""
    working = df[df["is_service_week"]].copy()
    # Deduplicate: week_index may be in feature_cols already
    meta_cols = [c for c in ["bus_id", "week_index"] if c not in feature_cols]
    cols_needed = feature_cols + [target_col] + meta_cols
    working = working[cols_needed].dropna()

    X = working[feature_cols].values
    y = working[target_col].values
    bus_ids = working["bus_id"].values.ravel()
    week_indices = working["week_index"].values.ravel()

    return X, y, bus_ids, week_indices, working.index


# ── Split strategies ───────────────────────────────────────────────────────

def split_group_kfold(X, y, bus_ids, n_splits=4):
    """GroupKFold by bus_id. Yields (train_idx, test_idx) arrays."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups=bus_ids))


def split_leave_buses_out(bus_ids, holdout=(10, 11, 12)):
    """Fixed holdout: train on most buses, test on holdout set."""
    holdout_set = set(holdout)
    train_idx = np.where(~np.isin(bus_ids, list(holdout_set)))[0]
    test_idx = np.where(np.isin(bus_ids, list(holdout_set)))[0]
    return train_idx, test_idx


def split_temporal(bus_ids, week_indices, train_frac=0.8):
    """80/20 temporal split within each bus."""
    train_idx, test_idx = [], []
    for bid in np.unique(bus_ids):
        positions = np.flatnonzero(bus_ids == bid)
        # Sort by week_index within this bus
        order = np.argsort(week_indices[positions])
        sorted_positions = positions[order]
        n_train = int(len(sorted_positions) * train_frac)
        train_idx.extend(sorted_positions[:n_train].tolist())
        test_idx.extend(sorted_positions[n_train:].tolist())
    return np.array(train_idx), np.array(test_idx)


# ── Run execution ──────────────────────────────────────────────────────────

def _next_run_number() -> int:
    """Find the next run number by scanning existing run dirs."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in RUNS_DIR.iterdir() if d.is_dir()]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[0]))
        except ValueError:
            continue
    return max(nums, default=0) + 1


def execute_run(model_name: str, target_type: str, df: pd.DataFrame = None):
    """Execute a single model run with full evaluation and output."""

    # ── Load data ──────────────────────────────────────────────────────
    if df is None:
        df = pd.read_csv(FEATURE_MATRIX_PATH, parse_dates=["week_start", "week_end"])

    target_col = TARGET_CUMULATIVE if target_type == "cumulative" else TARGET_DELTA
    feature_cols = get_feature_cols(df, target_type)
    X, y, bus_ids, week_indices, valid_idx = prepare_data(df, feature_cols, target_col)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target: {target_col} ({target_type})")
    print(f"Features: {feature_cols}")

    # ── Create run directory ───────────────────────────────────────────
    run_num = _next_run_number()
    run_name = f"{run_num:03d}_{model_name}_{target_type}"
    run_dir = RUNS_DIR / run_name
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun: {run_name}")
    print(f"Output: {run_dir}")

    # ── Save config ────────────────────────────────────────────────────
    pipeline = get_model(model_name)
    config = {
        "run_name": run_name,
        "run_number": run_num,
        "model_name": model_name,
        "target_type": target_type,
        "target_column": target_col,
        "dataset": ACTIVE_DATASET,
        "base_dataset": BASE_DATASET,
        "dataset_variant": DATASET_VARIANT,
        "noise_enabled": NOISE_ENABLED,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": int(X.shape[0]),
        "n_buses": int(len(np.unique(bus_ids))),
        "random_seed": RANDOM_SEED,
        "holdout_buses": [10, 11, 12],
        "temporal_train_frac": 0.8,
        "group_kfold_splits": 4,
        "model_params": {
            k: str(v) for k, v in pipeline.get_params().items()
            if not k.startswith("steps") and k != "memory" and k != "verbose"
        },
        "timestamp": datetime.now().isoformat(),
        "dropped_columns": DROP_COLS,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Evaluate on all 3 split strategies ─────────────────────────────
    all_metrics = {}

    # 1. GroupKFold CV
    print("\n  [1/3] GroupKFold CV (4 folds by bus_id)...")
    folds = split_group_kfold(X, y, bus_ids, n_splits=4)
    cv_preds = np.full_like(y, np.nan, dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_model = clone(pipeline)
        fold_model.fit(X[train_idx], y[train_idx])
        cv_preds[test_idx] = fold_model.predict(X[test_idx])
    cv_metrics = compute_metrics(y, cv_preds, prefix="cv")
    all_metrics.update(cv_metrics)
    print(f"    R2={cv_metrics['cv_R2']:.4f}, RMSE={cv_metrics['cv_RMSE']:.4f}, MAE={cv_metrics['cv_MAE']:.4f}")

    # 2. Leave-buses-out
    print("  [2/3] Leave-buses-out (buses 10-12 held out)...")
    holdout_train, holdout_test = split_leave_buses_out(bus_ids)
    holdout_model = clone(pipeline)
    holdout_model.fit(X[holdout_train], y[holdout_train])
    holdout_preds = holdout_model.predict(X[holdout_test])
    holdout_metrics = compute_metrics(y[holdout_test], holdout_preds, prefix="holdout")
    all_metrics.update(holdout_metrics)
    print(f"    R2={holdout_metrics['holdout_R2']:.4f}, RMSE={holdout_metrics['holdout_RMSE']:.4f}, MAE={holdout_metrics['holdout_MAE']:.4f}")

    # 3. Temporal split
    print("  [3/3] Temporal split (80/20 per bus)...")
    temp_train, temp_test = split_temporal(bus_ids, week_indices)
    temp_model = clone(pipeline)
    temp_model.fit(X[temp_train], y[temp_train])
    temp_preds = temp_model.predict(X[temp_test])
    temp_metrics = compute_metrics(y[temp_test], temp_preds, prefix="temporal")
    all_metrics.update(temp_metrics)
    print(f"    R2={temp_metrics['temporal_R2']:.4f}, RMSE={temp_metrics['temporal_RMSE']:.4f}, MAE={temp_metrics['temporal_MAE']:.4f}")

    # ── Save metrics ───────────────────────────────────────────────────
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({k: round(float(v), 6) for k, v in all_metrics.items()}, f, indent=2)

    # ── Save predictions (temporal split — the most meaningful) ────────
    pred_df = pd.DataFrame({
        "bus_id": bus_ids[temp_test],
        "week_index": week_indices[temp_test],
        "actual": y[temp_test],
        "predicted": temp_preds,
        "residual": temp_preds - y[temp_test],
        "split": "temporal_test",
    })
    # Also save CV predictions
    cv_pred_df = pd.DataFrame({
        "bus_id": bus_ids,
        "week_index": week_indices,
        "actual": y,
        "predicted": cv_preds,
        "residual": cv_preds - y,
        "split": "group_kfold_cv",
    })
    full_pred_df = pd.concat([cv_pred_df, pred_df], ignore_index=True)
    full_pred_df.to_csv(run_dir / "predictions.csv", index=False)

    # ── Generate figures ───────────────────────────────────────────────
    print("\n  Generating figures...")

    # Actual vs predicted (CV)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_actual_vs_predicted(y, cv_preds,
                            title=f"{model_name} — Actual vs Predicted (GroupKFold CV)\n{target_col}",
                            save_path=fig_dir / "actual_vs_predicted.png", ax=ax)
    plt.close()

    # Residuals (CV)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_residuals(y, cv_preds,
                   title=f"{model_name} — Residuals (GroupKFold CV)\n{target_col}",
                   save_path=fig_dir / "residuals.png", ax=ax)
    plt.close()

    # Per-bus trajectories (temporal split — shows forecast quality)
    traj_df = df.loc[valid_idx].copy()
    full_model = clone(pipeline)
    full_model.fit(X, y)
    all_preds = full_model.predict(X)
    plot_prediction_trajectories(
        traj_df.assign(predicted=all_preds),
        all_preds,
        bus_ids=sorted(df["bus_id"].unique()),
        target_col=target_col,
        title=f"{model_name} — Per-Bus Trajectories\n{target_col}",
        save_path=fig_dir / "per_bus_trajectories.png",
    )
    plt.close("all")

    # Model-specific: coefficients or feature importances
    if model_name == "linear_regression":
        _plot_coefficients(full_model, feature_cols, fig_dir)
    elif model_name == "random_forest":
        importances = full_model.named_steps["model"].feature_importances_
        plot_feature_importances(
            importances, feature_cols,
            title=f"Random Forest — Feature Importances\n{target_col}",
            save_path=fig_dir / "feature_importances.png",
        )
        plt.close()

    # ── Generate summary.md ────────────────────────────────────────────
    _write_summary(run_dir, config, all_metrics, feature_cols, model_name, target_col)

    # ── Update run log ─────────────────────────────────────────────────
    _update_run_log(run_name, model_name, target_type, all_metrics)

    print(f"\n  Run complete: {run_dir}")
    return run_dir, all_metrics


def _plot_coefficients(pipeline, feature_cols, fig_dir):
    """Plot linear model coefficients as horizontal bar chart."""
    model = pipeline.named_steps["model"]
    coefs = model.coef_
    idx = np.argsort(np.abs(coefs))[::-1]

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_cols) * 0.3)))
    colors = ["#d32f2f" if c < 0 else "#1976d2" for c in coefs[idx]]
    ax.barh(range(len(idx)), coefs[idx][::-1], color=colors[::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx][::-1], fontsize=8)
    ax.set_xlabel("Coefficient (scaled features)")
    ax.set_title("Linear Regression Coefficients")
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()


def _write_summary(run_dir, config, metrics, feature_cols, model_name, target_col):
    """Write a human-readable summary.md for the run."""
    lines = [
        f"# Run: {config['run_name']}",
        f"",
        f"**Date**: {config['timestamp'][:10]}",
        f"**Dataset**: {config['dataset']}",
        f"**Model**: {model_name}",
        f"**Target**: {target_col} ({config['target_type']})",
        f"**Samples**: {config['n_samples']}",
        f"**Features**: {config['n_features']}",
        f"",
        f"## Results",
        f"",
        f"| Split Strategy | R2 | RMSE | MAE | MAPE |",
        f"|---|---|---|---|---|",
    ]
    for prefix, label in [("cv", "GroupKFold CV"), ("holdout", "Leave-Buses-Out"), ("temporal", "Temporal 80/20")]:
        r2 = metrics.get(f"{prefix}_R2", float("nan"))
        rmse = metrics.get(f"{prefix}_RMSE", float("nan"))
        mae = metrics.get(f"{prefix}_MAE", float("nan"))
        mape = metrics.get(f"{prefix}_MAPE", float("nan"))
        lines.append(f"| {label} | {r2:.4f} | {rmse:.4f} | {mae:.4f} | {mape:.2f}% |")

    lines.extend([
        f"",
        f"## Features Used",
        f"",
    ])
    for col in feature_cols:
        lines.append(f"- `{col}`")

    lines.extend([
        f"",
        f"## Dropped Columns",
        f"",
    ])
    for col in config["dropped_columns"]:
        lines.append(f"- `{col}`")

    lines.extend([
        f"",
        f"## Files",
        f"",
        f"- `config.json` — full reproducible configuration",
        f"- `metrics.json` — all metric scores",
        f"- `predictions.csv` — actual vs predicted for every test sample",
        f"- `figures/actual_vs_predicted.png` — scatter plot",
        f"- `figures/residuals.png` — residual analysis",
        f"- `figures/per_bus_trajectories.png` — per-bus prediction curves",
    ])
    if model_name == "linear_regression":
        lines.append(f"- `figures/coefficients.png` — feature coefficient magnitudes")
    elif model_name == "random_forest":
        lines.append(f"- `figures/feature_importances.png` — top feature importances")

    with open(run_dir / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def _update_run_log(run_name, model_name, target_type, metrics):
    """Append a row to the run log CSV."""
    log_path = RUNS_DIR / "run_log.csv"
    row = {
        "run": run_name,
        "model": model_name,
        "target": target_type,
        "timestamp": datetime.now().isoformat(),
        **{k: round(float(v), 6) for k, v in metrics.items()},
    }
    row_df = pd.DataFrame([row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(log_path, index=False)


# ── Batch 1 entry point ───────────────────────────────────────────────────

def run_batch_1():
    """Run the first batch: Linear Regression + Random Forest on both targets."""
    df = pd.read_csv(FEATURE_MATRIX_PATH, parse_dates=["week_start", "week_end"])

    runs = [
        ("linear_regression", "cumulative"),
        ("linear_regression", "delta"),
        ("random_forest", "cumulative"),
        ("random_forest", "delta"),
    ]

    results = {}
    for model_name, target_type in runs:
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()} — {target_type.upper()}")
        print(f"{'='*60}")
        run_dir, metrics = execute_run(model_name, target_type, df=df)
        results[(model_name, target_type)] = metrics

    print(f"\n\n{'='*60}")
    print("BATCH 1 COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"\nAll runs saved to: {RUNS_DIR}")
    print(f"Run log: {RUNS_DIR / 'run_log.csv'}")
    print(f"\nResults overview:")
    for (model, target), m in results.items():
        print(f"\n  {model} / {target}:")
        for prefix in ["cv", "holdout", "temporal"]:
            r2 = m.get(f"{prefix}_R2", float("nan"))
            rmse = m.get(f"{prefix}_RMSE", float("nan"))
            print(f"    {prefix:10s}: R2={r2:.4f}, RMSE={rmse:.4f}")

    return results


if __name__ == "__main__":
    run_batch_1()
