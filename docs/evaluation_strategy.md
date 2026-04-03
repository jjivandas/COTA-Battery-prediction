# Evaluation Strategy

## Data Leakage Prevention

### For cumulative target (`avg_qloss`)
- Cumulative features (total distance, total cycles, etc.) are included — they are legitimate physical predictors of total degradation
- `avg_qloss_cycling` and `avg_qloss_calendar` are EXCLUDED as features — they are sub-components of the target (their sum ≈ avg_qloss), so including them would be circular

### For delta target (`delta_qloss`)
- ONLY weekly features used (no cumulative columns)
- Rationale: if predicting "how much degradation happened this week," including "total degradation so far" would leak information about where we are on the degradation curve
- `week_index` stays as an age proxy — older batteries genuinely degrade differently, this is physics not leakage

### Lag and rolling features
- `delta_qloss_lag1` uses the PREVIOUS week's value — past data, not future
- Rolling means use backward-looking windows only (`min_periods=1`, no forward fill)

---

## Three Split Strategies

### 1. GroupKFold (4 folds by bus_id) — Primary Model Selection

**Question it answers:** Can this model generalize to a bus it has never seen?

**How it works:**
```
Fold 1: Train on buses [4,5,6,7,8,9,10,11,12]  → Test on buses [1,2,3]
Fold 2: Train on buses [1,2,3,7,8,9,10,11,12]  → Test on buses [4,5,6]
Fold 3: Train on buses [1,2,3,4,5,6,10,11,12]  → Test on buses [7,8,9]
Fold 4: Train on buses [1,2,3,4,5,6,7,8,9]     → Test on buses [10,11,12]
```

**ML reasoning:** Each bus has its own degradation trajectory shaped by routes, driver behavior, and charging patterns. GroupKFold guarantees that during each fold, the test buses were completely unseen during training — not a single row from those buses appeared. This forces the model to learn generalizable patterns (temperature effects, usage intensity) rather than memorizing a specific bus's curve.

**Use case:** Deploying the model on a new bus (bus 13, 14, etc.) that wasn't in the training data. This is the most realistic evaluation for fleet-wide deployment.

**Limitation:** The model sees all time periods during training (early AND late weeks from the training buses). It doesn't test whether the model can predict the future for a known bus.

---

### 2. Temporal Split (80/20 per bus) — Forecasting Evaluation

**Question it answers:** Can this model predict what happens next?

**How it works:**
```
Bus 1: Train on weeks 1-115  → Test on weeks 116-143
Bus 2: Train on weeks 1-118  → Test on weeks 119-147
...each bus split independently at the 80% mark
```

**ML reasoning:** Battery degradation is non-stationary — degradation rates change as the battery ages (they typically accelerate). A model that fits the first 2 years perfectly might fail on year 3 if the degradation curve has shifted. The temporal split directly tests whether the model can extrapolate into the future.

**Use case:** Predicting "what will this bus's degradation look like 6 months from now?" This is the forecasting / maintenance planning use case.

**Limitation:** The model sees all 12 buses during training (just not their future weeks). It can't tell you if the model generalizes to a completely new bus.

---

### 3. Leave-Buses-Out (buses 10-12) — Final Honest Evaluation

**Question it answers:** Is our model selection process overfitting?

**How it works:**
```
Train on buses 1-9 (all weeks)  →  Test on buses 10-12 (all weeks)
Never touched during model selection or hyperparameter tuning.
```

**ML reasoning:** GroupKFold is used to pick the best model and tune hyperparameters. But since we're optimizing against GroupKFold scores, there's a subtle risk of overfitting to the CV procedure itself (choosing a model that happens to do well on these particular fold splits). The leave-out set is a completely untouched final exam — it tells you if your reported performance is honest.

**Analogy:** GroupKFold is your practice exam. Leave-buses-out is the real exam you don't see until test day.

**Limitation:** Only 3 buses in the test set — higher variance in the estimate.

---

## Summary

| Strategy | Tests | Strength | Blind spot |
|---|---|---|---|
| GroupKFold | Cross-bus generalization | Uses all data efficiently, tests unseen-bus performance | Doesn't test future prediction |
| Temporal 80/20 | Future prediction | Tests forecasting per bus | Model sees all buses during training |
| Leave-buses-out | Honest final score | Untouched during tuning, no optimism bias | Smaller test set (3 buses only) |

A strong model should perform well on **all three**. Patterns to watch for:
- High GroupKFold + low temporal → model can't forecast (bad for maintenance planning)
- High temporal + low GroupKFold → model can't generalize (bad for new buses)
- High GroupKFold + low leave-out → your model selection is overfitting the CV splits

---

## Metrics

| Metric | What it tells you |
|---|---|
| **RMSE** | Average prediction error in the same units as the target — penalizes large errors more |
| **MAE** | Average absolute error — more robust to outliers than RMSE |
| **R²** | Fraction of variance explained (1.0 = perfect, 0.0 = predicting the mean) |
| **MAPE** | Percentage error — useful for comparing across different scales |
