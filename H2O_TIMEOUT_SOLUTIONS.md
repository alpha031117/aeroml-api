# H2O AutoML Timeout Issue - Solutions Guide

## Problem Description

You're encountering this error:
```
AutoML was not able to build any model within a max runtime constraint of 120 seconds, you may want to increase this value before retrying.
```

This happens when H2O AutoML cannot complete training any models within the specified time limit.

## Root Causes

1. **Dataset too large** - Too many rows or columns
2. **Complex algorithms** - DeepLearning, StackedEnsemble, XGBoost take too long
3. **Too many models** - Default settings try to train too many models
4. **Cross-validation** - 5-fold CV doubles training time
5. **Resource constraints** - Limited CPU/memory
6. **Stopping criteria too strict** - Models don't converge quickly enough

## Solutions

### Solution 1: Quick Fix (Recommended)

Use the improved pipeline with optimized parameters:

```python
from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline

results = run_h2o_ml_pipeline(
    data_path="datasets/churn_data.csv",
    target_variable="Churn",
    max_runtime_secs=60,  # Reduced from 120s
    max_models=5,  # Reduced from 20
    exclude_algos=["DeepLearning", "StackedEnsemble", "XGBoost"],  # Exclude slow algorithms
    nfolds=2,  # Reduced from 5
    stopping_tolerance=0.01,  # More lenient
    stopping_rounds=2,  # Reduced from 3
    verbose=True
)
```

### Solution 2: Fallback Pipeline

Use the automatic fallback system:

```python
from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline_with_fallback

results = run_h2o_ml_pipeline_with_fallback(
    data_path="datasets/churn_data.csv",
    target_variable="Churn",
    initial_max_runtime=60,
    fallback_max_runtime=300,
    verbose=True
)
```

### Solution 3: Dataset Optimization

Reduce dataset size and complexity:

```python
from h2o_machine_learning_agent.h2o_ml_pipeline import optimize_dataset_for_automl

# Optimize dataset
optimized_df = optimize_dataset_for_automl(
    data_path="datasets/churn_data.csv",
    target_variable="Churn",
    max_rows=5000,  # Limit rows
    max_columns=20,  # Limit columns
    verbose=True
)

# Save and use optimized dataset
optimized_df.to_csv("datasets/optimized_churn_data.csv", index=False)
```

### Solution 4: Minimal Settings

For debugging or very fast results:

```python
results = run_h2o_ml_pipeline(
    data_path="datasets/churn_data.csv",
    target_variable="Churn",
    max_runtime_secs=30,
    max_models=2,
    exclude_algos=["DeepLearning", "StackedEnsemble", "XGBoost", "RandomForest", "GBM"],
    nfolds=1,  # No cross-validation
    balance_classes=False,
    return_predictions=False,
    return_leaderboard=False,
    return_performance=False,
    verbose=True
)
```

## Parameter Recommendations

### For Fast Training (30-60 seconds)
- `max_runtime_secs`: 30-60
- `max_models`: 2-5
- `exclude_algos`: ["DeepLearning", "StackedEnsemble", "XGBoost"]
- `nfolds`: 1-2
- `stopping_tolerance`: 0.01
- `stopping_rounds`: 2

### For Balanced Training (2-5 minutes)
- `max_runtime_secs`: 120-300
- `max_models`: 5-10
- `exclude_algos`: ["DeepLearning"]
- `nfolds`: 3
- `stopping_tolerance`: 0.001
- `stopping_rounds`: 3

### For Comprehensive Training (5+ minutes)
- `max_runtime_secs`: 300+
- `max_models`: 10-20
- `exclude_algos`: [] (include all)
- `nfolds`: 5
- `stopping_tolerance`: 0.001
- `stopping_rounds`: 3

## Testing Your Fix

Run the test script to verify solutions:

```bash
python test_timeout_fix.py
```

Or use the comprehensive example:

```bash
python h2o_machine_learning_agent/example_usage_fixed.py
```

## Production Recommendations

1. **Start with conservative settings** and increase gradually
2. **Use fallback pipeline** for robust error handling
3. **Monitor resource usage** during training
4. **Optimize dataset** before training if it's large
5. **Set appropriate timeouts** based on your infrastructure

## Common Issues and Fixes

### Issue: Still timing out with minimal settings
**Fix**: Check if your dataset is extremely large or has many categorical variables with high cardinality.

### Issue: Models not converging
**Fix**: Increase `stopping_tolerance` to 0.01 or 0.05, reduce `stopping_rounds` to 2.

### Issue: Memory errors
**Fix**: Reduce dataset size, exclude more algorithms, or use fewer cross-validation folds.

### Issue: Poor model performance
**Fix**: Gradually increase `max_models` and `max_runtime_secs` while monitoring performance.

## Example Usage

```python
# Quick start with timeout fix
from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline_with_fallback

results = run_h2o_ml_pipeline_with_fallback(
    data_path="your_data.csv",
    target_variable="your_target",
    initial_max_runtime=60,
    fallback_max_runtime=300
)

if results['status'] == 'completed':
    print(f"Model saved at: {results['model_path']}")
    print(f"Best model: {results['leaderboard']}")
else:
    print(f"Failed: {results['error']}")
```

This should resolve your H2O AutoML timeout issues!

## Session Management (New Feature)

The pipeline now supports keeping H2O sessions alive for leaderboard retrieval from other endpoints.

### Key Features

1. **Session Persistence**: H2O sessions remain active after pipeline completion
2. **Leaderboard Retrieval**: Get leaderboard from active sessions via API
3. **Session Monitoring**: Check session status and keep sessions alive
4. **Model Access**: Access all models in the active session

### New API Endpoints

- `GET /h2o-cluster-info` - Get H2O session status and cluster information
- `GET /h2o-active-models` - List all models in the active session
- `POST /h2o-keep-alive` - Keep H2O session alive
- `GET /h2o-leaderboard/{session_id}` - Get leaderboard from active session (enhanced)

### Example Usage

```python
# Run pipeline (session stays active)
results = run_h2o_ml_pipeline(...)

# Get session info
session_info = get_h2o_session_info()

# Keep session alive
keep_h2o_session_alive()

# Get leaderboard from active session
leaderboard = get_leaderboard_from_session("your_session_id")

# Get all active models
models = get_active_h2o_models()
```

### Important Notes

- **No Automatic Shutdown**: Sessions are not automatically closed
- **Manual Cleanup**: Call `shutdown_h2o()` when you want to close the session
- **Session Timeout**: Use keep-alive endpoint to prevent session timeout
- **Multiple Endpoints**: Multiple endpoints can access the same active session
