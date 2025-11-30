from fastapi import APIRouter, HTTPException, Depends
import uuid
import json
import shutil

from sqlalchemy.orm import Session

from app.helper.utils import SESSION_DATA_DIR, h2o_sessions
from app.db import crud
from app.db.database import get_db
from app.core.config import get_settings
from app.services.artifact_store import download_session_artifact, delete_artifact

h2o_utils_router = APIRouter(tags=["model-training-utils"])
settings = get_settings()


def _get_user_or_404(db: Session, user_id: str):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    user = crud.get_user(db, user_uuid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user, user_uuid


def _get_training_session_or_404(db: Session, session_id: str):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    training_session = crud.get_training_session(db, session_uuid)
    if not training_session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return training_session, session_uuid

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
def list_h2o_sessions(user_id: str, db: Session = Depends(get_db)):
    """
    List all available H2O training sessions.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing list of sessions with basic information
    """
    try:
        _, user_uuid = _get_user_or_404(db, user_id)
        training_sessions = crud.get_sessions_for_user(db, user_uuid)

        sessions_list = []
        for training_session in training_sessions:
            sessions_list.append({
                "session_id": str(training_session.session_id),
                "created_at": training_session.created_at.isoformat(),
                "status": training_session.status,
                "model_object_key": training_session.model_object_key,
                "session_object_key": training_session.session_object_key,
                "performance": training_session.performance,
            })
        
        return {
            "total_sessions": len(sessions_list),
            "sessions": sessions_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing sessions: {str(e)}"
        )

@h2o_utils_router.delete("/h2o-sessions/{session_id}")
def delete_h2o_session(session_id: str, user_id: str, db: Session = Depends(get_db)):
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
        _, user_uuid = _get_user_or_404(db, user_id)
        training_session, session_uuid = _get_training_session_or_404(db, session_id)

        if training_session.user_id != user_uuid:
            raise HTTPException(status_code=403, detail="Not authorized to delete this session")

        delete_artifact(settings.minio_model_bucket, training_session.model_object_key)
        delete_artifact(settings.minio_session_bucket, training_session.session_object_key)

        local_dir = SESSION_DATA_DIR / session_id
        if local_dir.exists():
            shutil.rmtree(local_dir)

        crud.delete_training_session(db, session_uuid)
        deleted_session = h2o_sessions.pop(session_id, {"status": training_session.status, "created_at": training_session.created_at.isoformat()})

        return {
            "message": f"Session {session_id} deleted successfully",
            "deleted_session": {
                "session_id": session_id,
                "status": deleted_session.get('status'),
                "created_at": deleted_session.get('created_at'),
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
def get_h2o_session_data_from_files(session_id: str, user_id: str, db: Session = Depends(get_db)):
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
        _, user_uuid = _get_user_or_404(db, user_id)
        training_session, _ = _get_training_session_or_404(db, session_id)

        if training_session.user_id != user_uuid:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")

        session_dir = SESSION_DATA_DIR / session_id
        
        primary_session_file = session_dir / "session_data.json"
        if not primary_session_file.exists():
            if training_session.session_object_key:
                try:
                    download_session_artifact(training_session.session_object_key, session_dir)
                except Exception as exc:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session artifacts not available in MinIO: {exc}"
                    )
            else:
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
            "data": session_data,
            "artifacts": {
                "model_object_key": training_session.model_object_key,
                "session_object_key": training_session.session_object_key,
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session data for {session_id}: {str(e)}"
        )

@h2o_utils_router.get("/h2o-session-data")
def list_h2o_session_data_directories(user_id: str, db: Session = Depends(get_db)):
    """
    List all available H2O session data directories.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing list of available session data directories
    """
    try:
        _, user_uuid = _get_user_or_404(db, user_id)
        training_sessions = crud.get_sessions_for_user(db, user_uuid)

        sessions = []
        for training_session in training_sessions:
            session_id = str(training_session.session_id)
            session_dir = SESSION_DATA_DIR / session_id

            available_files = []
            if session_dir.exists():
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
                "status": training_session.status,
                "created_at": training_session.created_at.isoformat(),
                "available_data": available_files,
                "data_count": len(available_files),
                "model_object_key": training_session.model_object_key,
                "session_object_key": training_session.session_object_key,
            })
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing session data directories: {str(e)}"
        )

