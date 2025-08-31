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
    verbose: bool = True,
    # New parameters for better AutoML configuration
    max_models: int = 10,
    exclude_algos: Optional[list] = None,
    nfolds: int = 3,
    balance_classes: bool = True,
    stopping_metric: str = "logloss",
    stopping_tolerance: float = 0.001,
    stopping_rounds: int = 3
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
            exclude_columns = [df.columns[0]]  # Exclude first column by default
        
        training_data = df.drop(columns=exclude_columns)
        
        # 6. Run the ML Agent
        if verbose:
            print("6. Running H2O ML Agent...")
        
        # Set default exclude algorithms if not provided
        if exclude_algos is None:
            exclude_algos = ["DeepLearning", "StackedEnsemble"]
        
        # Create enhanced user instructions with AutoML parameters
        enhanced_instructions = f"""
{user_instructions}

Please use the following AutoML parameters for faster training:
- max_models: {max_models}
- exclude_algos: {exclude_algos}
- nfolds: {nfolds}
- balance_classes: {balance_classes}
- stopping_metric: {stopping_metric}
- stopping_tolerance: {stopping_tolerance}
- stopping_rounds: {stopping_rounds}

Focus on faster algorithms like GLM, GBM, and Random Forest for quick results.
"""
        
        ml_agent.invoke_agent(
            data_raw=training_data,
            user_instructions=enhanced_instructions,
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
        
        # Initialize H2O if not already initialized
        try:
            h2o.init()
            results['h2o_session_info'] = get_h2o_session_info()
            if verbose:
                print("   H2O session initialized")
        except Exception as e:
            # H2O might already be initialized
            if verbose:
                print(f"   H2O session check: {e}")
        
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


def get_h2o_session_info():
    """
    Get information about the current H2O session without shutting it down.
    
    Returns:
    --------
    Dict[str, Any]
        Information about the current H2O session
    """
    try:
        cluster_info = h2o.cluster()
        return {
            'cluster_status': 'active',
            'cluster_id': cluster_info.cluster_id,
            'node_count': cluster_info.node_count,
            'total_cpu_count': cluster_info.total_cpu_count,
            'total_memory_size': cluster_info.total_memory_size,
            'free_memory_size': cluster_info.free_memory_size,
            'version': cluster_info.version
        }
    except Exception as e:
        return {
            'cluster_status': 'error',
            'error': str(e)
        }


def keep_h2o_session_alive():
    """
    Keep H2O session alive by performing a lightweight operation.
    This prevents the session from timing out.
    """
    try:
        # Perform a lightweight operation to keep session alive
        h2o.cluster().show_status()
        return True
    except Exception as e:
        print(f"Warning: Could not keep H2O session alive: {e}")
        return False


def run_h2o_ml_pipeline_with_fallback(
    data_path: str = "datasets/churn_data.csv",
    target_variable: str = "Churn",
    initial_max_runtime: int = 120,
    fallback_max_runtime: int = 300,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run H2O ML pipeline with automatic fallback to longer runtime if initial attempt fails.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    target_variable : str
        Name of the target variable for prediction
    initial_max_runtime : int
        Initial maximum runtime in seconds
    fallback_max_runtime : int
        Fallback maximum runtime in seconds if initial attempt fails
    verbose : bool
        Whether to print progress information
    **kwargs : dict
        Additional arguments to pass to run_h2o_ml_pipeline
    
    Returns:
    --------
    Dict[str, Any]
        Pipeline results
    """
    
    if verbose:
        print(f"=== Starting H2O ML Pipeline with Fallback ===")
        print(f"Initial runtime: {initial_max_runtime}s, Fallback runtime: {fallback_max_runtime}s")
    
    # First attempt with initial runtime
    results = run_h2o_ml_pipeline(
        data_path=data_path,
        target_variable=target_variable,
        max_runtime_secs=initial_max_runtime,
        verbose=verbose,
        **kwargs
    )
    
    # Check if the first attempt failed due to timeout
    if results['status'] == 'failed' and 'timeout' in results['error'].lower():
        if verbose:
            print(f"\n⚠️  Initial attempt failed due to timeout. Retrying with {fallback_max_runtime}s runtime...")
        
        # Second attempt with longer runtime and more conservative settings
        fallback_kwargs = kwargs.copy()
        fallback_kwargs.update({
            'max_runtime_secs': fallback_max_runtime,
            'max_models': 5,  # Reduce number of models
            'exclude_algos': ["DeepLearning", "StackedEnsemble", "XGBoost"],  # Exclude more algorithms
            'nfolds': 2,  # Reduce cross-validation folds
            'stopping_tolerance': 0.01,  # More lenient stopping
            'stopping_rounds': 2  # Fewer stopping rounds
        })
        
        results = run_h2o_ml_pipeline(
            data_path=data_path,
            target_variable=target_variable,
            verbose=verbose,
            **fallback_kwargs
        )
        
        if results['status'] == 'completed':
            if verbose:
                print("✅ Fallback attempt successful!")
        else:
            if verbose:
                print(f"❌ Fallback attempt also failed: {results['error']}")
    
    return results


def optimize_dataset_for_automl(
    data_path: str,
    target_variable: str,
    max_rows: int = 10000,
    max_columns: int = 50,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Optimize dataset for AutoML by reducing size and complexity.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    target_variable : str
        Name of the target variable
    max_rows : int
        Maximum number of rows to keep
    max_columns : int
        Maximum number of columns to keep
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    pd.DataFrame
        Optimized dataset
    """
    import pandas as pd
    
    if verbose:
        print(f"=== Optimizing Dataset for AutoML ===")
    
    # Load data
    df = pd.read_csv(data_path)
    
    if verbose:
        print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Sample rows if dataset is too large
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        if verbose:
            print(f"Sampled to {len(df)} rows")
    
    # Select most important columns if too many
    if len(df.columns) > max_columns:
        # Keep target variable and first few columns
        important_cols = [target_variable] + list(df.columns[:max_columns-1])
        df = df[important_cols]
        if verbose:
            print(f"Selected {len(df.columns)} most important columns")
    
    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        if verbose:
            print(f"Removed {len(constant_cols)} constant columns")
    
    # Remove high-cardinality categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_cardinality_cols = []
    for col in categorical_cols:
        if df[col].nunique() > 50:  # More than 50 unique values
            high_cardinality_cols.append(col)
    
    if high_cardinality_cols:
        df = df.drop(columns=high_cardinality_cols)
        if verbose:
            print(f"Removed {len(high_cardinality_cols)} high-cardinality columns")
    
    if verbose:
        print(f"Optimized dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def get_leaderboard_from_session(session_id: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Retrieve leaderboard from an active H2O session.
    
    Parameters:
    -----------
    session_id : str
        The session ID to retrieve leaderboard for
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    Dict[str, Any]
        Leaderboard data and session information
    """
    try:
        if verbose:
            print(f"=== Retrieving Leaderboard for Session {session_id} ===")
        
        # Keep session alive
        keep_h2o_session_alive()
        
        # Get session info
        session_info = get_h2o_session_info()
        
        # Try to get leaderboard from H2O
        try:
            # Get all models in the session
            models = h2o.ls()
            model_ids = [model for model in models if 'AutoML' in model]
            
            if not model_ids:
                return {
                    'session_id': session_id,
                    'status': 'no_models',
                    'message': 'No AutoML models found in current session',
                    'h2o_session_info': session_info
                }
            
            # Get the latest AutoML model
            latest_model_id = model_ids[-1]
            aml = h2o.get_model(latest_model_id)
            
            # Get leaderboard
            leaderboard_df = aml.leaderboard.as_data_frame()
            leaderboard_dict = leaderboard_df.to_dict()
            
            # Get performance metrics
            performance = aml.leader.model_performance()
            metrics = {
                'auc': performance.auc(),
                'logloss': performance.logloss(),
                'aucpr': performance.aucpr(),
                'mean_per_class_error': performance.mean_per_class_error(),
                'rmse': performance.rmse(),
                'mse': performance.mse(),
                'gini': performance.gini()
            }
            
            return {
                'session_id': session_id,
                'status': 'success',
                'leaderboard': leaderboard_dict,
                'best_model_id': aml.leader.model_id,
                'performance_metrics': metrics,
                'num_models': len(leaderboard_df),
                'h2o_session_info': session_info,
                'model_path': aml.leader.model_id
            }
            
        except Exception as e:
            return {
                'session_id': session_id,
                'status': 'error',
                'error': str(e),
                'message': 'Could not retrieve leaderboard from H2O session',
                'h2o_session_info': session_info
            }
            
    except Exception as e:
        return {
            'session_id': session_id,
            'status': 'failed',
            'error': str(e),
            'message': 'Failed to access H2O session'
        }


def get_active_h2o_models(verbose: bool = True) -> Dict[str, Any]:
    """
    Get all active models in the current H2O session.
    
    Parameters:
    -----------
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    Dict[str, Any]
        Information about active models
    """
    try:
        if verbose:
            print("=== Getting Active H2O Models ===")
        
        # Keep session alive
        keep_h2o_session_alive()
        
        # Get all models
        models = h2o.ls()
        
        # Filter AutoML models
        automl_models = [model for model in models if 'AutoML' in model]
        
        # Get model details
        model_details = []
        for model_id in automl_models:
            try:
                model = h2o.get_model(model_id)
                model_details.append({
                    'model_id': model_id,
                    'model_type': type(model).__name__,
                    'algorithm': getattr(model, 'algorithm', 'Unknown'),
                    'model_category': getattr(model, 'model_category', 'Unknown')
                })
            except Exception as e:
                model_details.append({
                    'model_id': model_id,
                    'error': str(e)
                })
        
        return {
            'status': 'success',
            'total_models': len(models),
            'automl_models': len(automl_models),
            'automl_model_details': model_details,
            'all_models': list(models),
            'h2o_session_info': get_h2o_session_info()
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'message': 'Failed to get active H2O models'
        }

