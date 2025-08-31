"""
Example usage of the improved H2O ML Pipeline with timeout solutions.

This example demonstrates how to handle AutoML timeout issues by:
1. Using optimized AutoML parameters
2. Using the fallback pipeline
3. Optimizing the dataset
4. Using conservative settings for faster training
"""

from h2o_ml_pipeline import (
    run_h2o_ml_pipeline,
    run_h2o_ml_pipeline_with_fallback,
    optimize_dataset_for_automl,
    shutdown_h2o,
    predict_with_model,
    evaluate_model_performance
)
import pandas as pd


def solve_timeout_issue_example():
    """
    Example showing how to solve the AutoML timeout issue.
    """
    print("=== Solving H2O AutoML Timeout Issue ===")
    
    # Solution 1: Use optimized parameters for faster training
    print("\n🔧 Solution 1: Optimized Parameters")
    results = run_h2o_ml_pipeline(
        data_path="datasets/churn_data.csv",
        target_variable="Churn",
        max_runtime_secs=60,  # Reduced from 120s
        max_models=5,  # Reduced number of models
        exclude_algos=["DeepLearning", "StackedEnsemble", "XGBoost"],  # Exclude slow algorithms
        nfolds=2,  # Reduce cross-validation
        stopping_tolerance=0.01,  # More lenient stopping
        stopping_rounds=2,  # Fewer stopping rounds
        verbose=True
    )
    
    if results['status'] == 'completed':
        print("✅ Solution 1 successful!")
        return results
    else:
        print(f"❌ Solution 1 failed: {results['error']}")
    
    # Solution 2: Use fallback pipeline
    print("\n🔄 Solution 2: Fallback Pipeline")
    results = run_h2o_ml_pipeline_with_fallback(
        data_path="datasets/churn_data.csv",
        target_variable="Churn",
        initial_max_runtime=60,
        fallback_max_runtime=180,
        verbose=True
    )
    
    if results['status'] == 'completed':
        print("✅ Solution 2 successful!")
        return results
    else:
        print(f"❌ Solution 2 failed: {results['error']}")
    
    # Solution 3: Optimize dataset first
    print("\n📊 Solution 3: Dataset Optimization")
    
    # Optimize the dataset
    optimized_df = optimize_dataset_for_automl(
        data_path="datasets/churn_data.csv",
        target_variable="Churn",
        max_rows=5000,  # Limit to 5000 rows
        max_columns=20,  # Limit to 20 columns
        verbose=True
    )
    
    # Save optimized dataset
    optimized_path = "datasets/optimized_churn_data.csv"
    optimized_df.to_csv(optimized_path, index=False)
    
    # Run pipeline on optimized dataset
    results = run_h2o_ml_pipeline(
        data_path=optimized_path,
        target_variable="Churn",
        max_runtime_secs=60,
        max_models=3,  # Very conservative
        exclude_algos=["DeepLearning", "StackedEnsemble", "XGBoost", "RandomForest"],
        nfolds=2,
        verbose=True
    )
    
    if results['status'] == 'completed':
        print("✅ Solution 3 successful!")
        return results
    else:
        print(f"❌ Solution 3 failed: {results['error']}")
    
    return results


def quick_test_example():
    """
    Quick test with minimal settings for debugging.
    """
    print("\n=== Quick Test with Minimal Settings ===")
    
    results = run_h2o_ml_pipeline(
        data_path="datasets/churn_data.csv",
        target_variable="Churn",
        max_runtime_secs=30,  # Very short runtime
        max_models=2,  # Only 2 models
        exclude_algos=["DeepLearning", "StackedEnsemble", "XGBoost", "RandomForest", "GBM"],
        nfolds=1,  # No cross-validation
        balance_classes=False,  # Disable class balancing
        return_predictions=False,
        return_leaderboard=False,
        return_performance=False,
        verbose=True
    )
    
    if results['status'] == 'completed':
        print("✅ Quick test successful!")
        print(f"Model saved at: {results['model_path']}")
    else:
        print(f"❌ Quick test failed: {results['error']}")
    
    return results


def production_ready_example():
    """
    Production-ready example with robust error handling.
    """
    print("\n=== Production-Ready Example ===")
    
    try:
        # Step 1: Try with standard settings
        results = run_h2o_ml_pipeline(
            data_path="datasets/churn_data.csv",
            target_variable="Churn",
            max_runtime_secs=120,
            max_models=10,
            exclude_algos=["DeepLearning"],
            nfolds=3,
            verbose=True
        )
        
        if results['status'] == 'completed':
            print("✅ Standard pipeline successful!")
            
            # Make predictions
            if results['model'] is not None and results['data'] is not None:
                predictions = predict_with_model(results['model'], results['data'])
                print(f"Generated {len(predictions)} predictions")
            
            # Evaluate performance
            if results['model'] is not None:
                metrics = evaluate_model_performance(results['model'])
                if metrics:
                    print(f"AUC: {metrics['auc']:.4f}")
                    print(f"LogLoss: {metrics['logloss']:.4f}")
            
            return results
        
        # Step 2: If standard fails, try fallback
        print("\n🔄 Standard pipeline failed, trying fallback...")
        results = run_h2o_ml_pipeline_with_fallback(
            data_path="datasets/churn_data.csv",
            target_variable="Churn",
            initial_max_runtime=60,
            fallback_max_runtime=300,
            verbose=True
        )
        
        if results['status'] == 'completed':
            print("✅ Fallback pipeline successful!")
            return results
        
        # Step 3: If all else fails, use minimal settings
        print("\n🔄 All attempts failed, using minimal settings...")
        results = quick_test_example()
        
        return results
        
    except Exception as e:
        print(f"❌ Production pipeline failed with exception: {e}")
        return {'status': 'failed', 'error': str(e)}


if __name__ == "__main__":
    # Run the timeout solution example
    results = solve_timeout_issue_example()
    
    if results['status'] == 'completed':
        print("\n🎉 Successfully solved the timeout issue!")
        
        # Show results
        if results['leaderboard'] is not None:
            print(f"\n📊 Leaderboard has {len(results['leaderboard'])} models")
        
        if results['model_path'] is not None:
            print(f"💾 Model saved at: {results['model_path']}")
        
        if results['performance'] is not None:
            print("📈 Model performance metrics available")
    
    else:
        print(f"\n❌ Could not solve timeout issue: {results['error']}")
        print("\nTrying quick test...")
        quick_test_example()
    
    # Cleanup
    shutdown_h2o()
