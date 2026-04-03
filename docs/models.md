# Model Selection Rationale

## Overview

We run 9 models and compare them rather than picking one upfront. The data decides which model wins. Each model brings a different assumption about the data — running all of them reveals which assumptions hold.

## The Models

### Linear Models — Start Simple

**Linear Regression**
- Assumes: target = weighted sum of features
- No tuning. If this gets R² > 0.90, the relationship is mostly linear and you don't need complex models.
- Acts as the baseline that every other model must beat.

**Ridge Regression**
- Same as linear but adds L2 penalty (shrinks all coefficients toward zero)
- Tuned parameter: `alpha` (penalty strength) — searched over [0.01, 0.1, 1.0, 10.0, 100.0]
- Why: our features are correlated (e.g. `weekly_distance` and `weekly_net_energy` move together). Without regularization, correlated features get wildly large coefficients that cancel each other. Ridge stabilizes this.

**Lasso Regression**
- Same as linear but adds L1 penalty (can drive coefficients to exactly zero)
- Tuned parameter: `alpha` — searched over [0.001, 0.01, 0.1, 1.0, 10.0]
- Why: acts as automatic feature selection. After training, any feature with coefficient = 0 is deemed unimportant. Useful for understanding which features actually matter.

### Tree-Based Models — Handle Nonlinearity

**Decision Tree**
- Splits data into regions using if-then rules (e.g. "if week_index > 50 and avg_batt_temp > 25, predict 0.15")
- Tuned: `max_depth` [5, 10, 15, None], `min_samples_leaf` [2, 5, 10]
- Fully interpretable — you can visualize the tree and see exactly why each prediction was made.
- Prone to overfitting if not constrained, which is why we tune depth and leaf size.

**Random Forest**
- Trains many decision trees on random subsets of data and features, averages their predictions
- Tuned: `n_estimators` [100, 200], `max_depth` [10, 20, None]
- Why: a single tree overfits. Averaging 100+ trees dramatically reduces variance. The "random" part (each tree sees different features) means trees make different errors that cancel out.

**Gradient Boosting**
- Trains trees sequentially — each new tree corrects the errors of the previous ones
- Tuned: `n_estimators` [100, 200], `learning_rate` [0.05, 0.1], `max_depth` [3, 5]
- Why: usually the top performer on tabular/structured data. The sequential error-correction is very efficient. Smaller trees with low learning rate tend to generalize best.

### Kernel-Based — Different Geometry

**SVR (Support Vector Regression)**
- Maps data into a higher-dimensional space using a kernel (RBF = radial basis function), then fits a linear model there
- Tuned: `C` [1.0, 10.0, 100.0], `epsilon` [0.01, 0.1]
- Why: completely different approach from trees. Can capture complex nonlinear relationships. Works well when the dataset is small (which ours is at ~1777 rows).
- Note: requires feature scaling (StandardScaler in the pipeline), since SVR is distance-based.

### Probabilistic Models — Predictions with Uncertainty

**Bayesian Ridge**
- Like Ridge regression but Bayesian — outputs a probability distribution over predictions, not just a point estimate
- No manual tuning (learns its own regularization strength from data)
- Why: gives you prediction uncertainty. Instead of "degradation will be 0.13 this week," it says "0.13 ± 0.02." Critical for maintenance planning where you need confidence levels.

**Gaussian Process**
- Full probabilistic model — assumes data comes from a multivariate Gaussian distribution
- Kernel: RBF (smooth patterns) + WhiteKernel (noise)
- Why: gives the richest uncertainty estimates. Confidence intervals naturally widen in regions with less training data. If the model hasn't seen a particular operating condition, it honestly says "I'm not sure."
- Limitation: scales as O(n³), so we subsample to 500 rows for training. Not practical for very large datasets, but fine for ~1777 rows.

## Hyperparameter Tuning

All tuning uses `GridSearchCV` with `GroupKFold` (4 folds by bus_id):
- For each hyperparameter combination, the model is trained on 3 folds and evaluated on the 4th
- The combination with the best average score across all 4 folds is selected
- This prevents overfitting to a specific train/test split

Models with no hyperparameters (Linear Regression, Bayesian Ridge, Gaussian Process) are just trained and evaluated directly.

## What to Expect

**Cumulative target (avg_qloss):**
- R² > 0.95 for most models — because loss mostly increases with time, even simple models capture this trend
- This doesn't mean the model is great — it means the target is easy. The real test is the delta target.

**Delta target (delta_qloss):**
- R² will be much lower (0.3-0.7 range is realistic)
- This is the meaningful result — it tells you how well we can predict week-to-week variation in degradation
- Feature importances from this target reveal which operational factors actually drive degradation

## Feature Scaling

SVR, Bayesian Ridge, and Gaussian Process are sensitive to feature scales (a column in kilometers vs a column in fractions would dominate differently). These models have `StandardScaler` in their pipeline (zero mean, unit variance). Tree-based models don't need scaling — they split on thresholds, not distances.

---

## Revised Approach: Iterative, Not Shotgun

The above describes every model we *could* run. But running all 9 at once is not professional practice. We take an iterative approach: start simple, understand results, let the data tell us what to try next.

### Columns Excluded from Modeling

Before training, we drop columns that are not useful as features:

| Column | Reason for exclusion |
|---|---|
| `week_start`, `week_end` | Date strings — temporal info already captured by `week_index` |
| `routes` | Categorical string (e.g. "route2,route1") — already encoded as `route_count` |
| `bus_id` | Group identifier, not a predictor — used for splitting, not as a feature |
| `is_service_week` | Rows with `is_service_week=False` are dropped entirely (no operational data that week) |
| `weekly_cycles` | Near-constant: 1725/1770 rows have value 7.0. No predictive variance. |
| `avg_qloss_cycling`, `avg_qloss_calendar` | Sub-components of `avg_qloss` target — including them is circular |
| `delta_qloss_cycling`, `delta_qloss_calendar` | Sub-components of `delta_qloss` target — same leakage issue |

### Run Batch 1: Baseline (Linear Regression + Random Forest)

Four runs total — two models, two targets:

| Run | Model | Target | Question |
|---|---|---|---|
| 001 | Linear Regression | cumulative (`avg_qloss`) | How much does a straight line explain total degradation? |
| 002 | Linear Regression | delta (`delta_qloss`) | Can we predict weekly degradation linearly? |
| 003 | Random Forest | cumulative (`avg_qloss`) | Are there nonlinear patterns the linear model missed? |
| 004 | Random Forest | delta (`delta_qloss`) | Nonlinear drivers of weekly degradation? |

**Why these two models:**
- Linear Regression gives interpretable coefficients — you see exactly how much each feature contributes
- Random Forest captures nonlinear relationships and interactions — comparing it to linear tells you whether complexity is needed

**What we learn from this batch:**
- If Linear ≈ Random Forest → relationship is mostly linear, no need for complex models
- If Random Forest >> Linear → nonlinear effects matter, investigate with feature importance plots
- If both are poor → features need rework, not models

### Run Output Structure

Each run is self-contained and reproducible:

```
results/mar-2026/runs/
├── 001_linear_regression_cumulative/
│   ├── config.json           # model params, feature list, split strategy, random seed
│   ├── metrics.json          # RMSE, MAE, R², MAPE for each split strategy
│   ├── predictions.csv       # bus_id, week_index, actual, predicted for every test sample
│   ├── figures/
│   │   ├── actual_vs_predicted.png
│   │   ├── residuals.png
│   │   ├── per_bus_trajectories.png
│   │   └── coefficients.png (linear) or feature_importances.png (RF)
│   └── summary.md            # human-readable results writeup
├── 002_linear_regression_delta/
│   └── ...
├── 003_random_forest_cumulative/
│   └── ...
├── 004_random_forest_delta/
│   └── ...
└── run_log.csv               # one row per run for quick comparison across all runs
```

Anyone on the team can:
1. Open `config.json` to see exactly what was run
2. Open `summary.md` for a plain-English explanation
3. Look at `figures/` to see the results visually
4. Re-run using the same config to reproduce results

### After Batch 1

Review results and decide:
- If linear is competitive → try Ridge/Lasso next (better regularization, feature selection)
- If RF is significantly better → try Gradient Boosting (usually outperforms RF)
- If delta target R² is very low for both → revisit features before adding model complexity
- If uncertainty matters for deployment → add Bayesian Ridge or Gaussian Process
