from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime
import json
from app.helper.utils import DATASETS_DIR, save_session_data_to_files, SESSION_DATA_DIR, h2o_sessions
from typing import Dict, Any

h2o_utils_router = APIRouter(tags=["model-training-utils"])

@h2o_utils_router.get("/h2o-ml-pipeline-status")
def h2o_ml_pipeline_status():
    """
    Check if H2O ML pipeline is available and ready to use.
    """
    try:
        # Try to import the required modules
        from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline
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

@h2o_utils_router.get("/h2o-sessions")
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

@h2o_utils_router.delete("/h2o-sessions/{session_id}")
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

@h2o_utils_router.get("/h2o-session-data/{session_id}")
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

@h2o_utils_router.get("/h2o-session-data")
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

# @h2o_utils_router.get("/h2o-cluster-info")
# def get_h2o_cluster_info():
#     """
#     Get information about the current H2O cluster status.
    
#     Returns:
#     --------
#     Dict[str, Any]
#         Dictionary containing H2O cluster information
#     """
#     try:
#         import h2o
        
#         # Check if H2O is running
#         try:
#             cluster_info = h2o.cluster_info()
#             return {
#                 "status": "running",
#                 "cluster_info": cluster_info,
#                 "message": "H2O cluster is running"
#             }
#         except Exception as e:
#             return {
#                 "status": "not_running",
#                 "error": str(e),
#                 "message": "H2O cluster is not running or not accessible"
#             }
            
#     except ImportError:
#         return {
#             "status": "h2o_not_available",
#             "error": "H2O library is not installed",
#             "message": "H2O library is not available in the current environment"
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error getting H2O cluster info: {str(e)}"
#         )
