"""Generate forecast-vs-actual plot showing the train/forecast divide.

For each bus: train model on first 80% of weeks, predict the last 20%.
Plot actual values across all weeks, predictions on the held-out portion,
and a vertical line marking the train/forecast boundary.

This visually shows whether the model can extrapolate into the future.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

# Allow running from project root: python scripts/make_forecast_plot.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURE_MATRIX_PATH, RESULTS_DIR, TARGET_CUMULATIVE
from src.modeling import (
    get_feature_cols, get_model, prepare_data, split_temporal,
)


def make_forecast_plot(model_name: str = "linear_regression",
                       target_type: str = "cumulative",
                       train_frac: float = 0.8):
    df = pd.read_csv(FEATURE_MATRIX_PATH, parse_dates=["week_start", "week_end"])
    target_col = TARGET_CUMULATIVE if target_type == "cumulative" else "delta_qloss"
    feat_cols = get_feature_cols(df, target_type)
    X, y, bus_ids, week_indices, valid_idx = prepare_data(df, feat_cols, target_col)

    train_idx, test_idx = split_temporal(bus_ids, week_indices, train_frac=train_frac)
    pipeline = clone(get_model(model_name))
    pipeline.fit(X[train_idx], y[train_idx])

    train_pred = pipeline.predict(X[train_idx])
    test_pred = pipeline.predict(X[test_idx])

    plot_df = pd.DataFrame({
        "bus_id": np.concatenate([bus_ids[train_idx], bus_ids[test_idx]]),
        "week_index": np.concatenate([week_indices[train_idx], week_indices[test_idx]]),
        "actual": np.concatenate([y[train_idx], y[test_idx]]),
        "predicted": np.concatenate([train_pred, test_pred]),
        "phase": np.concatenate([
            np.full(len(train_idx), "train"),
            np.full(len(test_idx), "forecast"),
        ]),
    })

    bus_list = sorted(plot_df["bus_id"].unique())
    n = len(bus_list)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3 * rows), squeeze=False)

    for idx, bid in enumerate(bus_list):
        ax = axes[idx // cols][idx % cols]
        bus_data = plot_df[plot_df["bus_id"] == bid].sort_values("week_index")
        train_data = bus_data[bus_data["phase"] == "train"]
        test_data = bus_data[bus_data["phase"] == "forecast"]

        boundary = train_data["week_index"].max() if len(train_data) > 0 else None

        ax.plot(bus_data["week_index"], bus_data["actual"],
                color="#1976d2", linewidth=1.5, label="Actual")
        ax.plot(train_data["week_index"], train_data["predicted"],
                color="#43a047", linewidth=1.2, linestyle="--", label="Model fit (train)")
        ax.plot(test_data["week_index"], test_data["predicted"],
                color="#d32f2f", linewidth=1.5, linestyle="--", label="Forecast (unseen)")

        if boundary is not None:
            ax.axvline(boundary, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
            ax.text(boundary, ax.get_ylim()[1] * 0.05, " Forecast →",
                    fontsize=8, color="#555", verticalalignment="bottom")

        ax.set_title(f"Bus {bid}", fontsize=10)
        ax.set_xlabel("Week", fontsize=8)
        ax.set_ylabel(target_col, fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle(
        f"Can the model predict the future? — {model_name}, target = {target_col}\n"
        f"Trained on first {int(train_frac*100)}% of weeks per bus, forecasting the rest",
        fontsize=13,
    )
    plt.tight_layout()

    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forecast_{model_name}_{target_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_forecast_plot("linear_regression", "cumulative")
    make_forecast_plot("random_forest", "cumulative")
