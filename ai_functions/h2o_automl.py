# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: h2o_ml_agent
# Time Created: 2025-03-08 00:51:05

def h2o_automl(
    data_raw,
    target: str = 'Churn',
    max_runtime_secs: int = 30,
    exclude_algos: list = ["DeepLearning"],
    balance_classes: bool = True,
    nfolds: int = -1,
    seed: int = 42,
    model_directory: str = r'c:\Users\alpha\Documents\Side Project\AI Agents\ai-model-pipeline\h2o_models/',
    log_path: str = r'c:\Users\alpha\Documents\Side Project\AI Agents\ai-model-pipeline\ai_functions/',
    enable_mlflow: bool = False,
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = "H2O AutoML",
    mlflow_run_name: str = None
):

    import h2o
    from h2o.automl import H2OAutoML
    import pandas as pd

    # Start H2O Cluster
    h2o.init()

    # Convert data_raw (pandas DataFrame) to H2OFrame
    data_h2o = h2o.H2OFrame(data_raw)

    # Convert target to factor for classification
    data_h2o[target] = data_h2o[target].asfactor()

    # Set up AutoML
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        exclude_algos=exclude_algos,
        balance_classes=balance_classes,
        nfolds=nfolds,
        seed=seed,
        stopping_metric="AUC",
        stopping_rounds=3,
        stopping_tolerance=0.001,
    )

    # Training frame parameters
    x = [col for col in data_h2o.columns if col != target]
    
    # Train the model
    aml.train(x=x, y=target, training_frame=data_h2o)

    # Save the model
    model_path = h2o.save_model(model=aml.leader, path=model_directory, force=True)

    # Retrieve the leaderboard as a DataFrame and convert to dict
    leaderboard_df = aml.leaderboard.as_data_frame()
    leaderboard_dict = leaderboard_df.to_dict(orient="records")

    # Select the best model
    best_model_id = aml.leader.model_id
    best_model_metrics = leaderboard_df.iloc[0].to_dict()

    # Prepare model results
    model_results = {
        "model_flavor": "H2O AutoML",
        "best_model_id": best_model_id,
        "metrics": best_model_metrics,
    }

    # Optional MLflow logging
    if enable_mlflow:
        import mlflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        with mlflow.start_run(run_name=mlflow_run_name):
            # Log metrics
            numeric_metrics = {k: v for k, v in best_model_metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(numeric_metrics)
            # Log the model
            mlflow.h2o.log_model(aml.leader, artifact_path="model")
    
    # Return output
    return {
        "leaderboard": leaderboard_dict,
        "best_model_id": best_model_id,
        "model_path": model_path,
        "model_results": model_results,
    }