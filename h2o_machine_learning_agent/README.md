# H2O Machine Learning Pipeline

This directory contains a Python function that converts the Jupyter notebook functionality into a reusable H2O machine learning pipeline.

## Files

- `h2o_ml_pipeline.py` - Main pipeline function and utility functions
- `example_usage.py` - Example usage demonstrations
- `h20_ml_agent.ipynb` - Original Jupyter notebook
- `README.md` - This documentation file

## Quick Start

### Basic Usage

```python
from h2o_ml_pipeline import run_h2o_ml_pipeline

# Run the pipeline with default settings
results = run_h2o_ml_pipeline(
    data_path="datasets/churn_data.csv",
    config_path="config/credentials.yml",
    target_variable="Churn"
)

# Check if successful
if results['status'] == 'completed':
    print("Pipeline completed successfully!")
    print(f"Best model AUC: {results['performance'].auc():.4f}")
else:
    print(f"Pipeline failed: {results['error']}")
```

### Advanced Usage

```python
# Run with custom settings
results = run_h2o_ml_pipeline(
    data_path="datasets/churn_data.csv",
    config_path="config/credentials.yml",
    target_variable="Churn",
    user_instructions="Please do classification on 'Churn' with focus on high precision.",
    model_name="gpt-4o-mini",
    max_runtime_secs=60,
    exclude_columns=["customerID"],
    return_model=True,
    return_predictions=True,
    return_leaderboard=True,
    return_performance=True,
    verbose=True
)
```

## Function Parameters

### `run_h2o_ml_pipeline()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | `"datasets/churn_data.csv"` | Path to the CSV data file |
| `config_path` | str | `"config/credentials.yml"` | Path to credentials configuration |
| `target_variable` | str | `"Churn"` | Name of the target variable |
| `user_instructions` | str | Default instruction | Instructions for the ML agent |
| `model_name` | str | `"gpt-4o-mini"` | Language model to use |
| `max_runtime_secs` | int | `30` | Maximum AutoML runtime |
| `log_enabled` | bool | `True` | Enable logging |
| `log_path` | str | `None` | Log files path |
| `model_directory` | str | `None` | Model save directory |
| `exclude_columns` | list | `["customerID"]` | Columns to exclude |
| `return_model` | bool | `True` | Return trained model |
| `return_predictions` | bool | `True` | Return predictions |
| `return_leaderboard` | bool | `True` | Return model leaderboard |
| `return_performance` | bool | `True` | Return performance metrics |
| `verbose` | bool | `True` | Print progress information |

## Return Values

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `status` | str | Pipeline status ('started', 'completed', 'failed') |
| `model` | H2O Model | Trained H2O model (if requested) |
| `predictions` | H2O Frame | Model predictions (if requested) |
| `leaderboard` | DataFrame | Model leaderboard (if requested) |
| `performance` | H2O ModelMetrics | Performance metrics (if requested) |
| `ml_agent` | H2OMLAgent | The ML agent instance |
| `model_path` | str | Path to saved model |
| `data` | DataFrame | Processed dataset |
| `error` | str | Error message (if failed) |

## Utility Functions

### `get_ml_recommendations(ml_agent, markdown=True)`
Get machine learning recommendations from the H2O ML agent.

### `get_h2o_training_function(ml_agent, markdown=True)`
Get the H2O training function from the ML agent.

### `predict_with_model(model, data, verbose=True)`
Make predictions using a trained H2O model.

### `evaluate_model_performance(model, verbose=True)`
Evaluate model performance and return key metrics.

### `shutdown_h2o()`
Shutdown H2O cluster.

## Example Usage

### Simple Example

```python
from h2o_ml_pipeline import run_h2o_ml_pipeline, shutdown_h2o

# Run pipeline
results = run_h2o_ml_pipeline(
    data_path="datasets/churn_data.csv",
    config_path="config/credentials.yml",
    target_variable="Churn"
)

# Display results
if results['status'] == 'completed':
    print(f"Best model AUC: {results['performance'].auc():.4f}")
    print(f"Top models: {len(results['leaderboard'])}")
    
    # Cleanup
    shutdown_h2o()
```

### Advanced Example

```python
from h2o_ml_pipeline import (
    run_h2o_ml_pipeline, 
    predict_with_model, 
    evaluate_model_performance,
    shutdown_h2o
)

# Run with custom settings
results = run_h2o_ml_pipeline(
    data_path="datasets/churn_data.csv",
    config_path="config/credentials.yml",
    target_variable="Churn",
    user_instructions="Focus on high precision classification",
    max_runtime_secs=60,
    verbose=True
)

if results['status'] == 'completed':
    # Make predictions
    predictions = predict_with_model(results['model'], results['data'])
    
    # Get detailed metrics
    metrics = evaluate_model_performance(results['model'])
    print(f"Gini: {metrics['gini']:.4f}")
    print(f"AUCPR: {metrics['aucpr']:.4f}")
    
    # Cleanup
    shutdown_h2o()
```

## Requirements

- Python 3.7+
- pandas
- h2o
- langchain-openai
- pyyaml
- aeroml_data_science_team

## Configuration

Create a `config/credentials.yml` file with your OpenAI API key:

```yaml
openai:
  OPENAI_API_KEY: "your-openai-api-key-here"
```

## Notes

- The function automatically handles H2O cluster initialization and shutdown
- Models are saved to the specified model directory
- Logs are saved to the specified log path
- The function includes comprehensive error handling and progress reporting
- All original notebook functionality has been preserved and enhanced

## Running Examples

To run the example usage:

```bash
cd h2o_machine_learning_agent
python example_usage.py
```

Or run the main pipeline directly:

```bash
python h2o_ml_pipeline.py
```
