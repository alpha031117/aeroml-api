"""
Example Usage of H2O Machine Learning Pipeline
=============================================

This file demonstrates how to use the H2O ML pipeline function
converted from the Jupyter notebook.

Author: AI Assistant
Date: 2024
"""

from h2o_ml_pipeline import (
    run_h2o_ml_pipeline, 
    get_ml_recommendations, 
    get_h2o_training_function,
    predict_with_model,
    evaluate_model_performance,
    shutdown_h2o
)


def simple_example():
    """
    Simple example of running the H2O ML pipeline.
    """
    print("=== Simple H2O ML Pipeline Example ===")
    
    # Run the pipeline with default settings
    results = run_h2o_ml_pipeline(
        data_path="datasets/churn_data.csv",
        config_path="config/credentials.yml",
        target_variable="Churn",
        verbose=True
    )
    
    # Check results
    if results['status'] == 'completed':
        print("\n✅ Pipeline completed successfully!")
        
        # Show top models
        if results['leaderboard'] is not None:
            print("\n🏆 Top 3 Models:")
            print(results['leaderboard'].head(3)[['model_id', 'auc', 'logloss']])
        
        # Show performance
        if results['performance'] is not None:
            print(f"\n📊 Model Performance:")
            print(f"   AUC: {results['performance'].auc():.4f}")
            print(f"   LogLoss: {results['performance'].logloss():.4f}")
        
        # Get recommendations
        if results['ml_agent'] is not None:
            print("\n💡 ML Recommendations:")
            recommendations = get_ml_recommendations(results['ml_agent'])
            print(recommendations)
        
        # Cleanup
        shutdown_h2o()
        
    else:
        print(f"❌ Pipeline failed: {results['error']}")


def advanced_example():
    """
    Advanced example with custom settings.
    """
    print("\n=== Advanced H2O ML Pipeline Example ===")
    
    # Run with custom settings
    results = run_h2o_ml_pipeline(
        data_path="datasets/churn_data.csv",
        config_path="config/credentials.yml",
        target_variable="Churn",
        user_instructions="Please do classification on 'Churn' with focus on high precision. Use a max runtime of 60 seconds.",
        model_name="gpt-4o-mini",
        max_runtime_secs=60,
        exclude_columns=["customerID"],
        return_model=True,
        return_predictions=True,
        return_leaderboard=True,
        return_performance=True,
        verbose=True
    )
    
    if results['status'] == 'completed':
        print("\n✅ Advanced pipeline completed!")
        
        # Make predictions on new data (using same data for demo)
        if results['model'] is not None and results['data'] is not None:
            print("\n🔮 Making predictions...")
            predictions = predict_with_model(results['model'], results['data'])
            if predictions is not None:
                print(f"   Generated {len(predictions)} predictions")
                print("   Sample predictions:")
                print(predictions.head())
        
        # Get detailed performance metrics
        if results['model'] is not None:
            print("\n📈 Detailed Performance Metrics:")
            metrics = evaluate_model_performance(results['model'])
            if metrics:
                print(f"   Gini: {metrics['gini']:.4f}")
                print(f"   AUCPR: {metrics['aucpr']:.4f}")
                print(f"   Mean Per Class Error: {metrics['mean_per_class_error']:.4f}")
        
        # Get training function
        if results['ml_agent'] is not None:
            print("\n🔧 H2O Training Function:")
            training_func = get_h2o_training_function(results['ml_agent'])
            print(training_func)
        
        shutdown_h2o()
        
    else:
        print(f"❌ Advanced pipeline failed: {results['error']}")


def minimal_example():
    """
    Minimal example with only essential components.
    """
    print("\n=== Minimal H2O ML Pipeline Example ===")
    
    # Run with minimal settings
    results = run_h2o_ml_pipeline(
        data_path="datasets/churn_data.csv",
        config_path="config/credentials.yml",
        target_variable="Churn",
        return_predictions=False,
        return_leaderboard=False,
        return_performance=False,
        verbose=False
    )
    
    if results['status'] == 'completed':
        print("✅ Minimal pipeline completed!")
        print(f"Model saved at: {results['model_path']}")
        shutdown_h2o()
    else:
        print(f"❌ Minimal pipeline failed: {results['error']}")


if __name__ == "__main__":
    # Run all examples
    simple_example()
    advanced_example()
    minimal_example()
