#!/usr/bin/env python3
"""
Example script demonstrating H2O session management without closing the session.

This script shows how to:
1. Run H2O ML pipeline
2. Keep the session alive
3. Retrieve leaderboard from active session
4. Get session information
5. Access active models
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from h2o_machine_learning_agent.h2o_ml_pipeline import (
    run_h2o_ml_pipeline,
    get_leaderboard_from_session,
    get_active_h2o_models,
    get_h2o_session_info,
    keep_h2o_session_alive,
    shutdown_h2o
)


def example_session_management():
    """
    Example of session management without closing H2O session.
    """
    print("=== H2O Session Management Example ===")
    
    # Step 1: Run H2O ML pipeline
    print("\n1. Running H2O ML Pipeline...")
    results = run_h2o_ml_pipeline(
        data_path="datasets/churn_data.csv",
        target_variable="Churn",
        max_runtime_secs=60,  # Reduced runtime
        max_models=5,  # Fewer models
        exclude_algos=["DeepLearning", "StackedEnsemble", "XGBoost"],
        nfolds=2,
        stopping_tolerance=0.01,
        stopping_rounds=2,
        verbose=True
    )
    
    if results['status'] != 'completed':
        print(f"❌ Pipeline failed: {results['error']}")
        return
    
    print("✅ Pipeline completed successfully!")
    
    # Step 2: Get session information
    print("\n2. Getting H2O Session Information...")
    session_info = get_h2o_session_info()
    print(f"Session Status: {session_info['cluster_status']}")
    print(f"Cluster ID: {session_info.get('cluster_id', 'N/A')}")
    print(f"Node Count: {session_info.get('node_count', 'N/A')}")
    
    # Step 3: Keep session alive
    print("\n3. Keeping Session Alive...")
    keep_alive_success = keep_h2o_session_alive()
    print(f"Keep-alive success: {keep_alive_success}")
    
    # Step 4: Get active models
    print("\n4. Getting Active Models...")
    models_info = get_active_h2o_models()
    print(f"Total models: {models_info.get('total_models', 0)}")
    print(f"AutoML models: {models_info.get('automl_models', 0)}")
    
    if models_info.get('automl_model_details'):
        print("AutoML Model Details:")
        for model in models_info['automl_model_details']:
            print(f"  - {model['model_id']} ({model.get('algorithm', 'Unknown')})")
    
    # Step 5: Retrieve leaderboard from active session
    print("\n5. Retrieving Leaderboard from Active Session...")
    session_id = "example_session_123"  # You can use any session ID
    leaderboard_result = get_leaderboard_from_session(session_id)
    
    if leaderboard_result['status'] == 'success':
        print(f"✅ Leaderboard retrieved successfully!")
        print(f"Number of models: {leaderboard_result['num_models']}")
        print(f"Best model ID: {leaderboard_result['best_model_id']}")
        
        # Show performance metrics
        metrics = leaderboard_result['performance_metrics']
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"LogLoss: {metrics['logloss']:.4f}")
        print(f"Gini: {metrics['gini']:.4f}")
    else:
        print(f"❌ Could not retrieve leaderboard: {leaderboard_result.get('message', 'Unknown error')}")
    
    # Step 6: Demonstrate session persistence
    print("\n6. Demonstrating Session Persistence...")
    print("Waiting 5 seconds to show session is still active...")
    time.sleep(5)
    
    # Check session again
    session_info_after = get_h2o_session_info()
    print(f"Session still active: {session_info_after['cluster_status'] == 'active'}")
    
    print("\n✅ Session management example completed!")
    print("Note: H2O session is still active and can be used by other endpoints")


def example_api_usage():
    """
    Example showing how the API endpoints would work with session management.
    """
    print("\n=== API Usage Example ===")
    print("This shows how the API endpoints would work:")
    
    print("\n1. Start H2O ML Pipeline:")
    print("   POST /run-h2o-ml-pipeline-advanced")
    print("   - Runs the pipeline and keeps session alive")
    
    print("\n2. Get Cluster Info:")
    print("   GET /h2o-cluster-info")
    print("   - Returns session status and cluster information")
    
    print("\n3. Get Active Models:")
    print("   GET /h2o-active-models")
    print("   - Lists all models in the active session")
    
    print("\n4. Keep Session Alive:")
    print("   POST /h2o-keep-alive")
    print("   - Prevents session timeout")
    
    print("\n5. Get Leaderboard:")
    print("   GET /h2o-leaderboard/{session_id}")
    print("   - Retrieves leaderboard from active session")
    
    print("\n6. List Sessions:")
    print("   GET /h2o-sessions")
    print("   - Lists all stored session information")


if __name__ == "__main__":
    try:
        # Run the session management example
        example_session_management()
        
        # Show API usage example
        example_api_usage()
        
        print("\n🎉 All examples completed successfully!")
        print("H2O session remains active for further use.")
        
        # Note: We don't call shutdown_h2o() here to keep the session alive
        print("To manually shutdown H2O session, call: shutdown_h2o()")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        # Cleanup on error
        shutdown_h2o()
