"""
AeroML API - FastAPI application for machine learning operations

This API provides endpoints for:
- Data source suggestions
- Dataset management
- H2O Machine Learning pipeline execution
- Model leaderboard retrieval
- Session management for H2O training runs

H2O ML Pipeline Endpoints:
- GET /run-h2o-ml-pipeline - Run basic H2O ML pipeline
- POST /run-h2o-ml-pipeline-advanced - Run advanced H2O ML pipeline with custom config
- GET /h2o-leaderboard/{session_id} - Get model leaderboard and ML recommendations from local files
- GET /h2o-ml-recommendations/{session_id} - Get ML recommendations from local files
- GET /h2o-sessions - List all available H2O training sessions
- DELETE /h2o-sessions/{session_id} - Delete a specific H2O training session
- GET /h2o-cluster-info - Get H2O cluster status information
- GET /h2o-ml-pipeline-status - Check H2O ML pipeline availability
- GET /h2o-session-data - List all available session data directories
- GET /h2o-session-data/{session_id} - Retrieve session data from local files

Each H2O training run generates a unique session ID that can be used to retrieve
the leaderboard and other results later. Session data including leaderboard metrics,
ML recommendations, and performance reports are automatically saved to local files
for easy retrieval without relying on H2O sessions.
"""

from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import subprocess
import shutil
import json
import asyncio
import sys
import os
import logging
from generate_LoRa_dataset import suggest_sources
from pathlib import Path
import pandas as pd
from utils import list_dataset_name, DATASETS_DIR
from typing import Optional

# Import the pipeline function
from h2o_machine_learning_agent.h2o_ml_pipeline import (
    run_h2o_ml_pipeline, 
    shutdown_h2o,
    get_leaderboard_from_session,
    get_active_h2o_models,
    get_h2o_session_info,
    keep_h2o_session_alive
)

# Add session management imports
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import json
from pathlib import Path

app = FastAPI(
    title="AeroML API",
    description="FastAPI application for dataset elicitation and H2O ML pipelines",
    version="1.0.0",
    openapi_tags=[
        {"name": "dataset-elicitation", "description": "Discover/prepare datasets & utilities"},
        {"name": "model-training", "description": "Run H2O pipelines, manage sessions & results"},
    ],
)

dataset_router = APIRouter(tags=["dataset-elicitation"])
h2o_router = APIRouter(tags=["model-training"])

# Global session storage (in production, use a proper database)
h2o_sessions: Dict[str, Dict[str, Any]] = {}

# Create directory for storing session data
SESSION_DATA_DIR = Path("session_data")
SESSION_DATA_DIR.mkdir(exist_ok=True)

def save_session_data_to_files(session_id: str, session_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Save session data to local files for easy retrieval.
    
    Parameters:
    -----------
    session_id : str
        The session ID
    session_data : Dict[str, Any]
        The session data to save
        
    Returns:
    --------
    Dict[str, str]
        Dictionary containing file paths for saved data
    """
    session_dir = SESSION_DATA_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    saved_files = {}
    
    try:
        # Save leaderboard data
        if session_data.get('leaderboard') is not None:
            leaderboard_file = session_dir / "leaderboard.json"
            if isinstance(session_data['leaderboard'], pd.DataFrame):
                leaderboard_data = session_data['leaderboard'].to_dict(orient='records')
            else:
                leaderboard_data = session_data['leaderboard']
            
            with open(leaderboard_file, 'w') as f:
                json.dump(leaderboard_data, f, indent=2, default=str)
            saved_files['leaderboard'] = str(leaderboard_file)
        
        # Save ML recommendations
        if session_data.get('ml_recommendations') is not None:
            recommendations_file = session_dir / "ml_recommendations.txt"
            with open(recommendations_file, 'w', encoding='utf-8') as f:
                f.write(session_data['ml_recommendations'])
            saved_files['ml_recommendations'] = str(recommendations_file)
        
        # Save performance metrics
        if session_data.get('performance') is not None:
            performance_file = session_dir / "performance.json"
            with open(performance_file, 'w') as f:
                json.dump(session_data['performance'], f, indent=2, default=str)
            saved_files['performance'] = str(performance_file)
        
        # Save complete session data
        session_file = session_dir / "session_data.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        saved_files['session_data'] = str(session_file)
        
        return saved_files
        
    except Exception as e:
        print(f"Error saving session data to files: {e}")
        return {}


# Event loop policy for compatibility (mostly safe fallback)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Allow CORS for specific origin
origins = [
    "http://localhost:3000",  # React frontend
    "http://127.0.0.1:80",   # If you want to allow backend to frontend communication
]

# Adding the CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the listed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@dataset_router.post("/suggest-sources")
async def suggest_sources_endpoint(request: Request):
    try:
        data = await request.json()
        if not data or "modelInput" not in data:
            return {"error": "Missing modelInput in request body", "status": 400}
        
        query = data.get("modelInput", "")
        if not query.strip():
            return {"error": "Query cannot be empty", "status": 400}
            
        sources = suggest_sources(query)
        return {"sources": sources, "status": 200}
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in request body", "status": 400}
    except Exception as e:
        error_message = str(e)
        print(f"Error in suggest-sources endpoint: {error_message}")  # Log the error
        return {
            "error": "An error occurred while processing your request",
            "details": error_message,
            "status": 500
        }

@dataset_router.get("/datasets")
async def get_dataset(filename: Optional[str] = None, limit: int = None, offset: int = 0):
    """
    Return a CSV (as JSON) for tabular display.
    - If `filename` is omitted, uses `list_dataset_name()` to choose one automatically.
    - Supports pagination with `limit` and `offset`.
    """
    # Choose a file when not specified
    if filename is None:
        filename = list_dataset_name()

    # Validate filename
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    file_path = DATASETS_DIR / filename

    # Security: ensure the resolved path is within the datasets directory
    try:
        file_path.resolve().relative_to(DATASETS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    try:
        # Load CSV
        df = pd.read_csv(str(file_path))

        total_rows = len(df)

        # Apply pagination
        if limit is not None:
            df = df.iloc[offset: offset + limit]

        # Build column meta (simple width; adjust as needed)
        columns = [{"field": col, "headerName": col, "width": 150} for col in df.columns]

        data = df.to_dict(orient="records")

        return {
            "data": data,
            "columns": columns,
            "totalRows": total_rows,
            "displayedRows": len(data),
            "offset": offset,
            "limit": limit,
            "filename": filename,
            "status": 200,
        }
    except Exception as e:
        # Fall back to 500 with details
        raise HTTPException(status_code=500, detail=f"Error processing {filename}: {e}")

@dataset_router.get("/run-stagehand")
def run_stagehand_script(request: Request):
    print("🔥 Starting subprocess...")
    def generate():
        context = None

        # Use full path to `npx.cmd` on Windows
        project_root = r"D:\alpha\Documents\PSM-AeroML\aeroml-api"  # <-- adjust
        npm_cmd = shutil.which("npm") or r"C:\Program Files\nodejs\npm.cmd"

        try:
            process = subprocess.Popen(
                [npm_cmd, "run", "stagehand", "--silent"],
                cwd=project_root,            # <-- IMPORTANT
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )

            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                decoded = line.strip()
                yield f"LOG: {decoded}\n"

                try:
                    parsed = json.loads(decoded)
                    if isinstance(parsed, dict) and parsed.get("type") == "final_output":
                        context = parsed.get("context")
                        yield f"\n\nFINAL_CONTEXT: {json.dumps(context, indent=2)}\n"
                except json.JSONDecodeError:
                    pass

            process.stdout.close()
            process.wait()

        except Exception as e:
            yield f"\n🔥 Error: {str(e)}\n"

    return StreamingResponse(generate(), media_type="text/plain")

@h2o_router.get("/model-training")
def model_training(request: Request):
    return {"message": "Model training started"}

@h2o_router.get("/h2o-ml-pipeline-status")
def h2o_ml_pipeline_status():
    """
    Check if H2O ML pipeline is available and ready to use.
    """
    try:
        # Try to import the required modules
        from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline, shutdown_h2o
        import h2o
        from langchain_openai import ChatOpenAI
        from aeroml_data_science_team.ml_agents import H2OMLAgent
        
        return {
            "status": "ready",
            "message": "H2O ML Pipeline is available and ready to use",
            "endpoints": {
                "simple": "GET /run-h2o-ml-pipeline",
                "advanced": "POST /run-h2o-ml-pipeline-advanced"
            },
            "features": [
                "Real-time execution logs",
                "Customizable parameters",
                "Model performance metrics",
                "Leaderboard generation",
                "Automatic H2O cluster management"
            ]
        }
    except ImportError as e:
        return {
            "status": "not_available",
            "error": f"Missing dependencies: {str(e)}",
            "message": "H2O ML Pipeline dependencies are not installed"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Error checking H2O ML Pipeline status"
        }

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

@h2o_router.get("/h2o-sessions")
def list_h2o_sessions():
    """
    List all available H2O training sessions.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing list of sessions with basic information
    """
    try:
        sessions_list = []
        for session_id, session_data in h2o_sessions.items():
            sessions_list.append({
                "session_id": session_id,
                "created_at": session_data['created_at'],
                "status": session_data['status'],
                "data_path": session_data.get('data_path'),
                "target_variable": session_data.get('target_variable'),
                "num_models": session_data.get('num_models', 0),
                "performance": session_data.get('performance')
            })
        
        # Sort by creation date (newest first)
        sessions_list.sort(key=lambda x: x['created_at'], reverse=True)
        
        return {
            "total_sessions": len(sessions_list),
            "sessions": sessions_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing sessions: {str(e)}"
        )

@h2o_router.delete("/h2o-sessions/{session_id}")
def delete_h2o_session(session_id: str):
    """
    Delete a specific H2O training session.
    
    Parameters:
    -----------
    session_id : str
        The session ID to delete
        
    Returns:
    --------
    Dict[str, Any]
        Confirmation message
    """
    try:
        if session_id not in h2o_sessions:
            raise HTTPException(
                status_code=404, 
                detail=f"Session {session_id} not found"
            )
        
        # Remove session from storage
        deleted_session = h2o_sessions.pop(session_id)
        
        return {
            "message": f"Session {session_id} deleted successfully",
            "deleted_session": {
                "session_id": session_id,
                "status": deleted_session['status'],
                "created_at": deleted_session['created_at']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session {session_id}: {str(e)}"
        )

@h2o_router.get("/h2o-session-data/{session_id}")
def get_h2o_session_data_from_files(session_id: str):
    """
    Retrieve H2O session data from local files.
    
    Parameters:
    -----------
    session_id : str
        The session ID to retrieve data for
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the session data from local files
    """
    try:
        session_dir = SESSION_DATA_DIR / session_id
        
        if not session_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Session data directory for {session_id} not found"
            )
        
        session_data = {}
        
        # Load leaderboard data
        leaderboard_file = session_dir / "leaderboard.json"
        if leaderboard_file.exists():
            with open(leaderboard_file, 'r') as f:
                session_data['leaderboard'] = json.load(f)
        
        # Load ML recommendations
        recommendations_file = session_dir / "ml_recommendations.txt"
        if recommendations_file.exists():
            with open(recommendations_file, 'r', encoding='utf-8') as f:
                session_data['ml_recommendations'] = f.read()
        
        # Load performance metrics
        performance_file = session_dir / "performance.json"
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                session_data['performance'] = json.load(f)
        
        # Load complete session data
        session_file = session_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_data['session_data'] = json.load(f)
        
        return {
            "session_id": session_id,
            "status": "success",
            "data_available": list(session_data.keys()),
            "data": session_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session data for {session_id}: {str(e)}"
        )

@h2o_router.get("/h2o-session-data")
def list_h2o_session_data_directories():
    """
    List all available H2O session data directories.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing list of available session data directories
    """
    try:
        if not SESSION_DATA_DIR.exists():
            return {
                "total_sessions": 0,
                "sessions": [],
                "message": "No session data directory found"
            }
        
        sessions = []
        for session_dir in SESSION_DATA_DIR.iterdir():
            if session_dir.is_dir():
                session_id = session_dir.name
                
                # Check what files are available
                available_files = []
                if (session_dir / "leaderboard.json").exists():
                    available_files.append("leaderboard")
                if (session_dir / "ml_recommendations.txt").exists():
                    available_files.append("ml_recommendations")
                if (session_dir / "performance.json").exists():
                    available_files.append("performance")
                if (session_dir / "session_data.json").exists():
                    available_files.append("session_data")
                
                sessions.append({
                    "session_id": session_id,
                    "available_data": available_files,
                    "data_count": len(available_files)
                })
        
        # Sort by session ID (newest first, assuming UUID format)
        sessions.sort(key=lambda x: x['session_id'], reverse=True)
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing session data directories: {str(e)}"
        )

@h2o_router.get("/h2o-cluster-info")
def get_h2o_cluster_info():
    """
    Get information about the current H2O cluster status.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing H2O cluster information
    """
    try:
        import h2o
        
        # Check if H2O is running
        try:
            cluster_info = h2o.cluster_info()
            return {
                "status": "running",
                "cluster_info": cluster_info,
                "message": "H2O cluster is running"
            }
        except Exception as e:
            return {
                "status": "not_running",
                "error": str(e),
                "message": "H2O cluster is not running or not accessible"
            }
            
    except ImportError:
        return {
            "status": "h2o_not_available",
            "error": "H2O library is not installed",
            "message": "H2O library is not available in the current environment"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting H2O cluster info: {str(e)}"
        )

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

app.include_router(dataset_router, prefix="")
app.include_router(h2o_router, prefix="")
