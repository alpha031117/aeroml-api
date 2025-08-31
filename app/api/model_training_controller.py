from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import uuid
from datetime import datetime
import json
from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline
from app.helper.utils import DATASETS_DIR, save_session_data_to_files, h2o_sessions
from typing import Dict, Any

h2o_router = APIRouter(tags=["model-training"])

@h2o_router.get("/model-training")
def model_training(request: Request):
    return {"message": "Model training started"}

@h2o_router.get("/run-h2o-ml-pipeline")
def run_h2o_ml_pipeline_endpoint(
    request: Request,
    data_path: str,
    target_variable: str,
    max_runtime_secs: int,
    model_name: str
):
    """
    Run H2O ML pipeline with real-time execution logs.
    This endpoint streams the execution progress and results.
    """
    print("🚀 Starting H2O ML Pipeline...")
    
    # Generate session ID for this training run
    session_id = str(uuid.uuid4())
    
    def generate():
        try:
            yield "LOG: Starting H2O Machine Learning Pipeline...\n"
            yield f"LOG: Session ID: {session_id}\n"
            
            # Run the pipeline with verbose output
            results = run_h2o_ml_pipeline(
                data_path="datasets/churn_data.csv",
                target_variable="Churn",
                max_runtime_secs=120,
                max_models=10,
                exclude_algos=["DeepLearning"],
                nfolds=3,
                verbose=True
            )
            
            # Extract ML recommendations if available
            ml_recommendations = None
            if results.get('ml_agent') is not None:
                try:
                    ml_recommendations = results['ml_agent'].get_recommended_ml_steps(markdown=False)
                    yield f"LOG: 📋 ML recommendations extracted\n"
                except Exception as e:
                    yield f"LOG: ⚠️ Could not extract ML recommendations: {e}\n"
        
            yield "LOG: Pipeline execution completed.\n"
            
            # Check results and provide summary
            if results['status'] == 'completed':
                yield "LOG: ✅ Pipeline completed successfully!\n"
                
                # Model performance summary
                if results['performance'] is not None:
                    auc = results['performance'].auc()
                    logloss = results['performance'].logloss()
                    yield f"LOG: 📊 Model Performance - AUC: {auc:.4f}, LogLoss: {logloss:.4f}\n"
                
                # Leaderboard summary
                if results['leaderboard'] is not None:
                    num_models = len(results['leaderboard'])
                    yield f"LOG: 🏆 Generated {num_models} models in leaderboard\n"
                    
                    # Show top 3 models
                    top_models = results['leaderboard'].head(3)
                    yield "LOG: Top 3 Models:\n"
                    for idx, row in top_models.iterrows():
                        model_id = row['model_id'][:50] + "..." if len(row['model_id']) > 50 else row['model_id']
                        yield f"LOG:   {idx+1}. {model_id} - AUC: {row['auc']:.4f}\n"
                
                # Model path
                if results['model_path']:
                    yield f"LOG: 💾 Model saved at: {results['model_path']}\n"
                
                # Final success message
                yield "LOG: 🎉 H2O ML Pipeline completed successfully!\n"
                
                # Store session data
                session_data = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "completed",
                    "data_path": data_path,
                    "target_variable": target_variable,
                    "max_runtime_secs": max_runtime_secs,
                    "model_name": model_name,
                    "model_path": results['model_path'],
                    "leaderboard": results['leaderboard'].to_dict(orient='records') if results['leaderboard'] is not None else None,
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": {
                        "auc": float(results['performance'].auc()) if results['performance'] else None,
                        "logloss": float(results['performance'].logloss()) if results['performance'] else None
                    } if results['performance'] else None,
                    "ml_recommendations": ml_recommendations
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)
                if saved_files:
                    yield f"LOG: 💾 Session data saved to local files:\n"
                    for file_type, file_path in saved_files.items():
                        yield f"LOG:   - {file_type}: {file_path}\n"
                
                # Return final results as JSON
                final_results = {
                    "status": "completed",
                    "message": "H2O ML Pipeline completed successfully",
                    "session_id": session_id,
                    "model_path": results['model_path'],
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": {
                        "auc": float(results['performance'].auc()) if results['performance'] else None,
                        "logloss": float(results['performance'].logloss()) if results['performance'] else None
                    } if results['performance'] else None,
                    "saved_files": saved_files,
                    "ml_recommendations_available": ml_recommendations is not None
                }
                
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                
            else:
                yield f"LOG: ❌ Pipeline failed: {results['error']}\n"
                
                # Store failed session data
                session_data = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "failed",
                    "data_path": data_path,
                    "target_variable": target_variable,
                    "max_runtime_secs": max_runtime_secs,
                    "model_name": model_name,
                    "error": results['error']
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save failed session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)
                
                final_results = {
                    "status": "failed",
                    "error": results['error'],
                    "message": "H2O ML Pipeline failed",
                    "session_id": session_id
                }
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
            
            # Cleanup
            # try:
            #     shutdown_h2o()
            #     yield "LOG: 🧹 H2O cluster shutdown successfully\n"
            # except Exception as e:
            #     yield f"LOG: ⚠️ Warning: Could not shutdown H2O cluster: {e}\n"
                
        except Exception as e:
            error_msg = f"🔥 Error in H2O ML Pipeline: {str(e)}"
            yield f"LOG: {error_msg}\n"
            
            # Store error session data
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "error",
                "data_path": data_path,
                "target_variable": target_variable,
                "max_runtime_secs": max_runtime_secs,
                "model_name": model_name,
                "error": str(e)
            }
            
            # Store in global session storage
            h2o_sessions[session_id] = session_data
            
            # Save error session data to local files
            saved_files = save_session_data_to_files(session_id, session_data)
            
            final_results = {
                "status": "error",
                "error": str(e),
                "message": "H2O ML Pipeline encountered an error",
                "session_id": session_id
            }
            yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@h2o_router.post("/run-h2o-ml-pipeline-advanced")
async def run_h2o_ml_pipeline_advanced_endpoint(request: Request):
    """
    Run H2O ML pipeline with advanced configuration via POST request.
    Accepts JSON configuration and returns real-time execution logs.
    """
    try:
        data = await request.json()
        
        # Extract parameters with defaults
        config = {
            "data_path": data.get("data_path", "datasets/churn_data.csv"),
            "target_variable": data.get("target_variable", "Churn"),
            "max_runtime_secs": data.get("max_runtime_secs", 30),
            "model_name": data.get("model_name", "gpt-4o-mini"),
            "user_instructions": data.get("user_instructions", ""),
            "exclude_columns": data.get("exclude_columns", ["customerID"]),
            "return_predictions": data.get("return_predictions", True),
            "return_leaderboard": data.get("return_leaderboard", True),
            "return_performance": data.get("return_performance", True)
        }
        
        # Use default instructions if not provided
        if not config["user_instructions"]:
            config["user_instructions"] = f"Please do classification on '{config['target_variable']}'. Use a max runtime of {config['max_runtime_secs']} seconds."
        
    except json.JSONDecodeError:
        # Return error for invalid JSON
        return {"error": "Invalid JSON in request body", "status": 400}
    except Exception as e:
        return {"error": f"Error parsing request: {str(e)}", "status": 400}
    
    # Generate session ID for this training run
    session_id = str(uuid.uuid4())
    
    print(f"🚀 Starting Advanced H2O ML Pipeline with config: {config}")
    
    def generate():
        try:
            # Import the pipeline function
            from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline, shutdown_h2o
            
            yield "LOG: Starting Advanced H2O Machine Learning Pipeline...\n"
            yield f"LOG: Session ID: {session_id}\n"
            yield f"LOG: Configuration - Data: {config['data_path']}, Target: {config['target_variable']}, Runtime: {config['max_runtime_secs']}s\n"
            
            # Run the pipeline with custom configuration
            results = run_h2o_ml_pipeline(
                data_path=config["data_path"],
                config_path="config/credentials.yml",
                target_variable=config["target_variable"],
                user_instructions=config["user_instructions"],
                model_name=config["model_name"],
                max_runtime_secs=config["max_runtime_secs"],
                exclude_columns=config["exclude_columns"],
                log_enabled=True,
                return_model=True,
                return_predictions=config["return_predictions"],
                return_leaderboard=config["return_leaderboard"],
                return_performance=config["return_performance"],
                verbose=True
            )
            
            # Extract ML recommendations if available
            ml_recommendations = None
            if results.get('ml_agent') is not None:
                try:
                    ml_recommendations = results['ml_agent'].get_recommended_ml_steps(markdown=False)
                    yield f"LOG: 📋 ML recommendations extracted\n"
                except Exception as e:
                    yield f"LOG: ⚠️ Could not extract ML recommendations: {e}\n"
            
            yield "LOG: Pipeline execution completed.\n"
            
            # Check results and provide summary
            if results['status'] == 'completed':
                yield "LOG: ✅ Advanced Pipeline completed successfully!\n"
                
                # Model performance summary
                if results['performance'] is not None:
                    auc = results['performance'].auc()
                    logloss = results['performance'].logloss()
                    yield f"LOG: 📊 Model Performance - AUC: {auc:.4f}, LogLoss: {logloss:.4f}\n"
                
                # Leaderboard summary
                if results['leaderboard'] is not None:
                    num_models = len(results['leaderboard'])
                    yield f"LOG: 🏆 Generated {num_models} models in leaderboard\n"
                    
                    # Show top 3 models
                    top_models = results['leaderboard'].head(3)
                    yield "LOG: Top 3 Models:\n"
                    for idx, row in top_models.iterrows():
                        model_id = row['model_id'][:50] + "..." if len(row['model_id']) > 50 else row['model_id']
                        yield f"LOG:   {idx+1}. {model_id} - AUC: {row['auc']:.4f}\n"
                
                # Model path
                if results['model_path']:
                    yield f"LOG: 💾 Model saved at: {results['model_path']}\n"
                
                # Final success message
                yield "LOG: 🎉 Advanced H2O ML Pipeline completed successfully!\n"
                
                # Store session data
                session_data = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "completed",
                    "data_path": config['data_path'],
                    "target_variable": config['target_variable'],
                    "max_runtime_secs": config['max_runtime_secs'],
                    "model_name": config['model_name'],
                    "user_instructions": config['user_instructions'],
                    "exclude_columns": config['exclude_columns'],
                    "model_path": results['model_path'],
                    "leaderboard": results['leaderboard'].to_dict(orient='records') if results['leaderboard'] is not None else None,
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": {
                        "auc": float(results['performance'].auc()) if results['performance'] else None,
                        "logloss": float(results['performance'].logloss()) if results['performance'] else None
                    } if results['performance'] else None,
                    "ml_recommendations": ml_recommendations
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)
                if saved_files:
                    yield f"LOG: 💾 Session data saved to local files:\n"
                    for file_type, file_path in saved_files.items():
                        yield f"LOG:   - {file_type}: {file_path}\n"
                
                # Return final results as JSON
                final_results = {
                    "status": "completed",
                    "message": "Advanced H2O ML Pipeline completed successfully",
                    "session_id": session_id,
                    "config": config,
                    "model_path": results['model_path'],
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": {
                        "auc": float(results['performance'].auc()) if results['performance'] else None,
                        "logloss": float(results['performance'].logloss()) if results['performance'] else None
                    } if results['performance'] else None,
                    "saved_files": saved_files,
                    "ml_recommendations_available": ml_recommendations is not None
                }
                
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                
            else:
                yield f"LOG: ❌ Advanced Pipeline failed: {results['error']}\n"
                
                # Store failed session data
                session_data = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "failed",
                    "data_path": config['data_path'],
                    "target_variable": config['target_variable'],
                    "max_runtime_secs": config['max_runtime_secs'],
                    "model_name": config['model_name'],
                    "user_instructions": config['user_instructions'],
                    "exclude_columns": config['exclude_columns'],
                    "error": results['error']
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save failed session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)
                
                final_results = {
                    "status": "failed",
                    "error": results['error'],
                    "message": "Advanced H2O ML Pipeline failed",
                    "session_id": session_id,
                    "config": config
                }
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
            
            # Cleanup
            try:
                shutdown_h2o()
                yield "LOG: 🧹 H2O cluster shutdown successfully\n"
            except Exception as e:
                yield f"LOG: ⚠️ Warning: Could not shutdown H2O cluster: {e}\n"
                
        except Exception as e:
            error_msg = f"🔥 Error in Advanced H2O ML Pipeline: {str(e)}"
            yield f"LOG: {error_msg}\n"
            
            # Store error session data
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "error",
                "data_path": config.get('data_path'),
                "target_variable": config.get('target_variable'),
                "max_runtime_secs": config.get('max_runtime_secs'),
                "model_name": config.get('model_name'),
                "user_instructions": config.get('user_instructions'),
                "exclude_columns": config.get('exclude_columns'),
                "error": str(e)
            }
            
            # Store in global session storage
            h2o_sessions[session_id] = session_data
            
            # Save error session data to local files
            saved_files = save_session_data_to_files(session_id, session_data)
            
            final_results = {
                "status": "error",
                "error": str(e),
                "message": "Advanced H2O ML Pipeline encountered an error",
                "session_id": session_id,
                "config": config
            }
            yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@h2o_router.get("/h2o-leaderboard/{session_id}")
def get_h2o_leaderboard(session_id: str):
    """
    Get the model leaderboard and ML recommendations for a specific H2O training session.
    Prioritizes local file storage over active H2O sessions for faster retrieval.
    
    Parameters:
    -----------
    session_id : str
        The session ID from a previous H2O ML pipeline run
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the leaderboard, ML recommendations, and session information
    """
    try:
        # First try to get data from local files (fastest)
        session_dir = SESSION_DATA_DIR / session_id
        
        if session_dir.exists():
            session_data = {}
            
            # Load leaderboard data from local file
            leaderboard_file = session_dir / "leaderboard.json"
            if leaderboard_file.exists():
                with open(leaderboard_file, 'r') as f:
                    session_data['leaderboard'] = json.load(f)
            
            # Load ML recommendations from local file
            recommendations_file = session_dir / "ml_recommendations.txt"
            if recommendations_file.exists():
                with open(recommendations_file, 'r', encoding='utf-8') as f:
                    session_data['ml_recommendations'] = f.read()
            
            # Load performance metrics from local file
            performance_file = session_dir / "performance.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    session_data['performance'] = json.load(f)
            
            # Load complete session data from local file
            session_file = session_dir / "session_data.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    complete_session_data = json.load(f)
                    
                    # Extract key information
                    session_data.update({
                        "session_id": complete_session_data.get('session_id'),
                        "created_at": complete_session_data.get('created_at'),
                        "status": complete_session_data.get('status'),
                        "data_path": complete_session_data.get('data_path'),
                        "target_variable": complete_session_data.get('target_variable'),
                        "max_runtime_secs": complete_session_data.get('max_runtime_secs'),
                        "model_name": complete_session_data.get('model_name'),
                        "model_path": complete_session_data.get('model_path'),
                        "num_models": complete_session_data.get('num_models', 0)
                    })
            
            # Check if we have leaderboard data
            if session_data.get('leaderboard'):
                return {
                    "session_id": session_id,
                    "status": "success",
                    "source": "local_files",
                    "created_at": session_data.get('created_at'),
                    "data_path": session_data.get('data_path'),
                    "target_variable": session_data.get('target_variable'),
                    "max_runtime_secs": session_data.get('max_runtime_secs'),
                    "model_name": session_data.get('model_name'),
                    "model_path": session_data.get('model_path'),
                    "num_models": session_data.get('num_models', 0),
                    "performance": session_data.get('performance'),
                    "leaderboard": session_data['leaderboard'],
                    "ml_recommendations": session_data.get('ml_recommendations'),
                    "data_available": list(session_data.keys())
                }
            else:
                return {
                    "session_id": session_id,
                    "status": "partial_data",
                    "source": "local_files",
                    "message": "Session data found but leaderboard not available",
                    "data_available": list(session_data.keys()),
                    "ml_recommendations": session_data.get('ml_recommendations')
                }
        
        # Fallback: Try to get leaderboard from active H2O session
        try:
            leaderboard_result = get_leaderboard_from_session(session_id)
            
            if leaderboard_result['status'] == 'success':
                # Successfully retrieved from active session
                return {
                    "session_id": session_id,
                    "status": "success",
                    "source": "active_h2o_session",
                    "leaderboard": leaderboard_result['leaderboard'],
                    "best_model_id": leaderboard_result['best_model_id'],
                    "performance_metrics": leaderboard_result['performance_metrics'],
                    "num_models": leaderboard_result['num_models'],
                    "h2o_session_info": leaderboard_result['h2o_session_info']
                }
        except Exception as e:
            # Log the error but continue to next fallback
            print(f"Warning: Could not retrieve from active H2O session: {e}")
        
        # Final fallback: Check stored session data in memory
        if session_id in h2o_sessions:
            session_data = h2o_sessions[session_id]
            
            # Check if session completed successfully
            if session_data['status'] == 'completed':
                return {
                    "session_id": session_id,
                    "status": "success",
                    "source": "memory_session_data",
                    "created_at": session_data['created_at'],
                    "data_path": session_data['data_path'],
                    "target_variable": session_data['target_variable'],
                    "max_runtime_secs": session_data['max_runtime_secs'],
                    "model_name": session_data['model_name'],
                    "model_path": session_data['model_path'],
                    "num_models": session_data.get('num_models', 0),
                    "performance": session_data.get('performance'),
                    "leaderboard": session_data.get('leaderboard'),
                    "ml_recommendations": session_data.get('ml_recommendations')
                }
            else:
                return {
                    "session_id": session_id,
                    "status": session_data['status'],
                    "source": "memory_session_data",
                    "error": session_data.get('error', 'Session did not complete successfully'),
                    "message": f"Session {session_id} status: {session_data['status']}"
                }
        
        # If not found anywhere, return error
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found in local files, active H2O session, or memory"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving leaderboard for session {session_id}: {str(e)}"
        )

@h2o_router.get("/h2o-ml-recommendations/{session_id}")
def get_h2o_ml_recommendations(session_id: str):
    """
    Get ML recommendations for a specific H2O training session from local files.
    
    Parameters:
    -----------
    session_id : str
        The session ID from a previous H2O ML pipeline run
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the ML recommendations and session information
    """
    try:
        # Try to get ML recommendations from local files first
        session_dir = SESSION_DATA_DIR / session_id
        
        if session_dir.exists():
            # Load ML recommendations from local file
            recommendations_file = session_dir / "ml_recommendations.txt"
            if recommendations_file.exists():
                with open(recommendations_file, 'r', encoding='utf-8') as f:
                    ml_recommendations = f.read()
                
                # Load session metadata
                session_data = {}
                session_file = session_dir / "session_data.json"
                if session_file.exists():
                    with open(session_file, 'r') as f:
                        complete_session_data = json.load(f)
                        session_data.update({
                            "session_id": complete_session_data.get('session_id'),
                            "created_at": complete_session_data.get('created_at'),
                            "status": complete_session_data.get('status'),
                            "data_path": complete_session_data.get('data_path'),
                            "target_variable": complete_session_data.get('target_variable'),
                            "max_runtime_secs": complete_session_data.get('max_runtime_secs'),
                            "model_name": complete_session_data.get('model_name')
                        })
                
                return {
                    "session_id": session_id,
                    "status": "success",
                    "source": "local_files",
                    "ml_recommendations": ml_recommendations,
                    "recommendations_length": len(ml_recommendations),
                    "created_at": session_data.get('created_at'),
                    "data_path": session_data.get('data_path'),
                    "target_variable": session_data.get('target_variable'),
                    "model_name": session_data.get('model_name')
                }
            else:
                return {
                    "session_id": session_id,
                    "status": "not_found",
                    "source": "local_files",
                    "message": "ML recommendations file not found for this session"
                }
        
        # Fallback: Check memory session data
        if session_id in h2o_sessions:
            session_data = h2o_sessions[session_id]
            
            if session_data.get('ml_recommendations'):
                return {
                    "session_id": session_id,
                    "status": "success",
                    "source": "memory_session_data",
                    "ml_recommendations": session_data['ml_recommendations'],
                    "recommendations_length": len(session_data['ml_recommendations']),
                    "created_at": session_data.get('created_at'),
                    "data_path": session_data.get('data_path'),
                    "target_variable": session_data.get('target_variable'),
                    "model_name": session_data.get('model_name')
                }
            else:
                return {
                    "session_id": session_id,
                    "status": "not_found",
                    "source": "memory_session_data",
                    "message": "ML recommendations not available in memory for this session"
                }
        
        # If not found anywhere, return error
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found in local files or memory"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving ML recommendations for session {session_id}: {str(e)}"
        )


