# Debugging Log

Issues encountered during implementation and how they were resolved.

---

## Bug 1: NumPy version incompatibility (system anaconda)

**Symptom**: Every `python -m src.*` command printed walls of warnings:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
AttributeError: _ARRAY_API not found
```

**Cause**: System anaconda had NumPy 2.x but packages like `numexpr` and `bottleneck` (pandas optional deps) were compiled against NumPy 1.x.

**Fix**: Created a project-level virtual environment (`venv/`) with clean installs from `requirements.txt`. All dependencies compile against the same NumPy version.

**Lesson**: Always use a project venv, never rely on system/anaconda packages.

---

## Bug 2: bus_id type mismatch (string vs int)

**Symptom**: `bus_id` was saved as string `"001"` in code but read back from CSV as integer `1`. The modeling code checked for holdout buses using `{"010", "011", "012"}` which would never match integer values.

**Cause**: `data_processing.py` set `bus_id = bus_id` (a string like "001"). After CSV round-trip, pandas parsed it as int. The modeling code used string comparisons.

**Fix**: 
- Changed `data_processing.py` to store `bus_id` as `int(bus_id)` explicitly
- Changed `modeling.py` holdout check to `{10, 11, 12}`

**Lesson**: Be explicit about types. CSV round-trips lose type information — don't rely on implicit string preservation.

---

## Bug 3: Duplicate `week_index` column causing 3D array (the big one)

**Symptom**: `StandardScaler` crashed with `ValueError: Found array with dim 3, while dim <= 2 is required` during the temporal split evaluation.

**Investigation**:
1. First suspected `split_temporal` was returning 2D index arrays — tested in isolation, worked fine with 1D output
2. Then suspected pycache serving stale bytecode — cleared `__pycache__`, still failed
3. Finally traced the actual shapes at runtime:
   ```
   week_indices shape: (1739, 2)  ← should be (1739,)
   temp_train shape: (1387, 2)    ← 2D because week_indices was 2D
   X[temp_train] shape: (1387, 2, 33)  ← 3D! StandardScaler can't handle this
   ```

**Root cause**: In `prepare_data()`, the line:
```python
cols_needed = feature_cols + [target_col, "bus_id", "week_index"]
```
`week_index` was already in `feature_cols` (it's used as an age proxy). Adding it again created a DataFrame with **two** `week_index` columns. When we did `working["week_index"].values`, pandas returned both columns as a `(n, 2)` array instead of `(n,)`.

This 2D `week_indices` propagated into `split_temporal`, which returned 2D index arrays, which made `X[temp_train]` produce a 3D array.

**Why GroupKFold and Leave-Buses-Out worked**: They split on `bus_ids`, not `week_indices`. `bus_id` was NOT in `feature_cols`, so it wasn't duplicated.

**Fix**: Deduplicate metadata columns before building `cols_needed`:
```python
meta_cols = [c for c in ["bus_id", "week_index"] if c not in feature_cols]
cols_needed = feature_cols + [target_col] + meta_cols
```
Also added `.ravel()` to `bus_ids` and `week_indices` as a safety net.

**Lesson**: When building column lists by concatenation, always check for duplicates. A DataFrame silently allows duplicate column names, and `.values` on a duplicated column returns all copies as extra dimensions.
