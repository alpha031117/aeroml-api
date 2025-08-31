import os
import yaml
import pandas as pd
import h2o
from langchain_openai import ChatOpenAI
from aeroml_data_science_team.ml_agents import H2OMLAgent
from typing import Optional, Dict, Any, Tuple
import warnings

def run_h2o_ml_pipeline(
    data_path: str = "datasets/churn_data.csv",
    config_path: str = "config/credentials.yml",
    target_variable: str = "Churn",
    user_instructions: str = "Please do classification on 'Churn'. Use a max runtime of 30 seconds.",
    model_name: str = "gpt-4o-mini",
    max_runtime_secs: int = 30,
    log_enabled: bool = True,
    log_path: Optional[str] = None,
    model_directory: Optional[str] = None,
    exclude_columns: Optional[list] = None,
    return_model: bool = True,
    return_predictions: bool = True,
    return_leaderboard: bool = True,
    return_performance: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a complete H2O machine learning pipeline including data loading, model training,
    evaluation, and prediction.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    config_path : str
        Path to the credentials configuration file
    target_variable : str
        Name of the target variable for prediction
    user_instructions : str
        Instructions for the ML agent
    model_name : str
        Name of the language model to use
    max_runtime_secs : int
        Maximum runtime for AutoML training in seconds
    log_enabled : bool
        Whether to enable logging
    log_path : str, optional
        Path for log files (defaults to "ai_functions/")
    model_directory : str, optional
        Directory for saving models (defaults to "h2o_models/")
    exclude_columns : list, optional
        Columns to exclude from the dataset
    return_model : bool
        Whether to return the trained model
    return_predictions : bool
        Whether to return predictions
    return_leaderboard : bool
        Whether to return the model leaderboard
    return_performance : bool
        Whether to return model performance metrics
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the pipeline results including:
        - 'model': Trained H2O model (if return_model=True)
        - 'predictions': Model predictions (if return_predictions=True)
        - 'leaderboard': Model leaderboard (if return_leaderboard=True)
        - 'performance': Model performance metrics (if return_performance=True)
        - 'ml_agent': The H2O ML agent instance
        - 'model_path': Path to the saved model
        - 'data': The processed dataset
        - 'status': Pipeline execution status
    """
    
    results = {
        'status': 'started',
        'model': None,
        'predictions': None,
        'leaderboard': None,
        'performance': None,
        'ml_agent': None,
        'model_path': None,
        'data': None,
        'error': None
    }
    
    try:
        if verbose:
            print("=== H2O Machine Learning Pipeline Started ===")
        
        # 1. Load and prepare data
        if verbose:
            print("1. Loading data...")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        results['data'] = df
        
        if verbose:
            print(f"   Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 2. Setup configuration and environment
        if verbose:
            print("2. Setting up configuration...")
        
        # Set default paths if not provided
        if log_path is None:
            log_path = os.path.join(os.getcwd(), "ai_functions/")
        if model_directory is None:
            model_directory = os.path.join(os.getcwd(), "h2o_models/")
        
        # 3. Initialize language model
        if verbose:
            print("3. Initializing language model...")
        
        llm = ChatOpenAI(model=model_name)
        
        # 4. Create H2O ML Agent
        if verbose:
            print("4. Creating H2O ML Agent...")
        
        ml_agent = H2OMLAgent(
            model=llm,
            log=log_enabled,
            log_path=log_path,
            model_directory=model_directory,
        )
        results['ml_agent'] = ml_agent
        
        # 5. Prepare data for training
        if verbose:
            print("5. Preparing data for training...")
        
        # Remove excluded columns
        if exclude_columns is None:
            exclude_columns = ["customerID"]  # Default exclusion
        
        training_data = df.drop(columns=exclude_columns)
        
        # 6. Run the ML Agent
        if verbose:
            print("6. Running H2O ML Agent...")
        
        ml_agent.invoke_agent(
            data_raw=training_data,
            user_instructions=user_instructions,
            target_variable=target_variable
        )
        
        # 7. Get results
        if verbose:
            print("7. Collecting results...")
        
        # Get leaderboard
        if return_leaderboard:
            try:
                leaderboard = ml_agent.get_leaderboard()
                results['leaderboard'] = leaderboard
                if verbose:
                    print(f"   Leaderboard retrieved with {len(leaderboard)} models")
            except Exception as e:
                warnings.warn(f"Could not retrieve leaderboard: {e}")
        
        # 8. Initialize H2O and load model
        if verbose:
            print("8. Initializing H2O and loading model...")
        
        h2o.init()
        
        # Get model path and load model
        try:
            model_path = ml_agent.get_model_path()
            results['model_path'] = model_path
            
            if return_model:
                model = h2o.load_model(model_path)
                results['model'] = model
                if verbose:
                    print(f"   Model loaded from: {model_path}")
        except Exception as e:
            warnings.warn(f"Could not load model: {e}")
        
        # 9. Get model performance
        if return_performance and results['model'] is not None:
            try:
                performance = results['model'].model_performance()
                results['performance'] = performance
                if verbose:
                    print("   Model performance metrics retrieved")
            except Exception as e:
                warnings.warn(f"Could not get model performance: {e}")
        
        # 10. Make predictions
        if return_predictions and results['model'] is not None:
            try:
                predictions = results['model'].predict(h2o.H2OFrame(df))
                results['predictions'] = predictions
                if verbose:
                    print("   Predictions generated")
            except Exception as e:
                warnings.warn(f"Could not generate predictions: {e}")
        
        # 11. Get additional information
        if results['model'] is not None:
            try:
                # Get base models for stacked ensemble
                base_models = results['model'].get_params().get("base_models", [])
                results['base_models'] = base_models
                if verbose:
                    print(f"   Base models in ensemble: {len(base_models)}")
            except Exception as e:
                warnings.warn(f"Could not get base models: {e}")
        
        results['status'] = 'completed'
        
        if verbose:
            print("=== H2O Machine Learning Pipeline Completed Successfully ===")
        
        return results
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        if verbose:
            print(f"=== Pipeline Failed: {e} ===")
        return results


def get_ml_recommendations(ml_agent: H2OMLAgent, markdown: bool = True) -> str:
    """
    Get machine learning recommendations from the H2O ML agent.
    
    Parameters:
    -----------
    ml_agent : H2OMLAgent
        The H2O ML agent instance
    markdown : bool
        Whether to return recommendations in markdown format
    
    Returns:
    --------
    str
        Machine learning recommendations
    """
    try:
        return ml_agent.get_recommended_ml_steps(markdown=markdown)
    except Exception as e:
        return f"Could not retrieve recommendations: {e}"


def get_h2o_training_function(ml_agent: H2OMLAgent, markdown: bool = True) -> str:
    """
    Get the H2O training function from the ML agent.
    
    Parameters:
    -----------
    ml_agent : H2OMLAgent
        The H2O ML agent instance
    markdown : bool
        Whether to return function in markdown format
    
    Returns:
    --------
    str
        H2O training function
    """
    try:
        return ml_agent.get_h2o_train_function(markdown=markdown)
    except Exception as e:
        return f"Could not retrieve training function: {e}"


def predict_with_model(model, data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Make predictions using a trained H2O model.
    
    Parameters:
    -----------
    model
        Trained H2O model
    data : pd.DataFrame
        Data to make predictions on
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    pd.DataFrame
        Predictions dataframe
    """
    try:
        if verbose:
            print("Making predictions...")
        
        predictions = model.predict(h2o.H2OFrame(data))
        return predictions.as_data_frame()
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None


def evaluate_model_performance(model, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate model performance and return key metrics.
    
    Parameters:
    -----------
    model
        Trained H2O model
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    Dict[str, Any]
        Performance metrics
    """
    try:
        if verbose:
            print("Evaluating model performance...")
        
        performance = model.model_performance()
        
        # Extract key metrics
        metrics = {
            'auc': performance.auc(),
            'logloss': performance.logloss(),
            'aucpr': performance.aucpr(),
            'mean_per_class_error': performance.mean_per_class_error(),
            'rmse': performance.rmse(),
            'mse': performance.mse(),
            'gini': performance.gini(),
            'confusion_matrix': performance.confusion_matrix().table
        }
        
        if verbose:
            print(f"   AUC: {metrics['auc']:.4f}")
            print(f"   LogLoss: {metrics['logloss']:.4f}")
            print(f"   RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    except Exception as e:
        print(f"Error evaluating model performance: {e}")
        return None


def shutdown_h2o():
    """
    Shutdown H2O cluster.
    """
    try:
        h2o.cluster().shutdown()
        print("H2O cluster shutdown successfully")
    except Exception as e:
        print(f"Error shutting down H2O cluster: {e}")

