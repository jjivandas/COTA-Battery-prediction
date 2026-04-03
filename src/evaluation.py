"""Evaluation metrics and visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from src.config import FIGURES_DIR


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """Compute RMSE, MAE, R-squared, MAPE."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = np.array(y_true)[mask], np.array(y_pred)[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    nonzero = np.abs(y_true) > 1e-10
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = np.nan

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}RMSE": rmse,
        f"{p}MAE": mae,
        f"{p}R2": r2,
        f"{p}MAPE": mape,
    }


def plot_actual_vs_predicted(y_true, y_pred, title: str = "Actual vs Predicted",
                              save_path: Path | None = None, ax=None):
    """Scatter plot of actual vs predicted values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_residuals(y_true, y_pred, title: str = "Residuals",
                   save_path: Path | None = None, ax=None):
    """Residual plot."""
    residuals = np.array(y_pred) - np.array(y_true)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_prediction_trajectories(df: pd.DataFrame, y_pred, bus_ids=None,
                                  target_col: str = "avg_qloss",
                                  title: str = "Prediction Trajectories",
                                  save_path: Path | None = None):
    """Plot actual vs predicted trajectories per bus."""
    plot_df = df.copy()
    plot_df["predicted"] = y_pred

    if bus_ids is None:
        bus_ids = sorted(plot_df["bus_id"].unique())[:6]

    n = len(bus_ids)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, bid in enumerate(bus_ids):
        ax = axes[idx // cols][idx % cols]
        bus_data = plot_df[plot_df["bus_id"] == bid].sort_values("week_index")
        ax.plot(bus_data["week_index"], bus_data[target_col], "b-", label="Actual", linewidth=1.5)
        ax.plot(bus_data["week_index"], bus_data["predicted"], "r--", label="Predicted", linewidth=1.5)
        ax.set_title(f"Bus {bid}")
        ax.set_xlabel("Week")
        ax.set_ylabel(target_col)
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importances(importances, feature_names, top_n: int = 15,
                              title: str = "Feature Importances",
                              save_path: Path | None = None):
    """Bar chart of top feature importances."""
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), importances[idx][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "R2",
                           title: str = "Model Comparison",
                           save_path: Path | None = None):
    """Bar chart comparing models on a given metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(results_df))
    ax.bar(x, results_df[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_prediction_intervals(df, y_pred, y_std, bus_ids=None,
                               target_col="avg_qloss",
                               title="Predictions with Uncertainty",
                               save_path=None):
    """Plot predictions with +/- 2 std confidence intervals."""
    plot_df = df.copy()
    plot_df["predicted"] = y_pred
    plot_df["std"] = y_std

    if bus_ids is None:
        bus_ids = sorted(plot_df["bus_id"].unique())[:4]

    n = len(bus_ids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, bid in enumerate(bus_ids):
        ax = axes[0][idx]
        bus_data = plot_df[plot_df["bus_id"] == bid].sort_values("week_index")
        weeks = bus_data["week_index"]
        ax.plot(weeks, bus_data[target_col], "b-", label="Actual")
        ax.plot(weeks, bus_data["predicted"], "r--", label="Predicted")
        ax.fill_between(
            weeks,
            bus_data["predicted"] - 2 * bus_data["std"],
            bus_data["predicted"] + 2 * bus_data["std"],
            alpha=0.2, color="red", label="95% CI",
        )
        ax.set_title(f"Bus {bid}")
        ax.legend(fontsize=7)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
