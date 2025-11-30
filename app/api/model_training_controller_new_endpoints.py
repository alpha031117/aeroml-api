# ============================================================================
# MinIO Artifact Retrieval Endpoints
# These endpoints should be added to the h2o_router in model_training_controller.py
# ============================================================================

# @h2o_router.get("/artifacts/list")
def list_user_artifacts_endpoint(user_id: str, db: Session = Depends(get_db)):
    """
    List all model and session artifacts available for the authenticated user in MinIO.

    Parameters:
    -----------
    user_id : str
        The user ID (UUID format)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing lists of available model and session artifacts with metadata
    """
    try:
        _, user_uuid = _get_user_or_404(db, user_id)

        artifacts = list_user_artifacts(user_uuid)

        return {
            "status": "success",
            "user_id": str(user_uuid),
            "total_models": len(artifacts["models"]),
            "total_sessions": len(artifacts["sessions"]),
            "models": artifacts["models"],
            "sessions": artifacts["sessions"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing artifacts: {str(e)}"
        )


# @h2o_router.get("/artifacts/model/{session_id}")
def get_model_artifact_info(session_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Get information about a trained model artifact in MinIO.

    Parameters:
    -----------
    session_id : str
        The session ID
    user_id : str
        The user ID (must be the owner of the session)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing model artifact information and metadata from the database
    """
    try:
        training_session, _, _, _ = _get_session_context(db, session_id, user_id)

        if not training_session.model_object_key:
            raise HTTPException(
                status_code=404,
                detail=f"No model artifact found for session {session_id}. "
                       f"The training session may have failed or not yet completed."
            )

        return {
            "status": "success",
            "session_id": session_id,
            "model_object_key": training_session.model_object_key,
            "training_status": training_session.status,
            "created_at": training_session.created_at.isoformat() if training_session.created_at else None,
            "metadata": training_session.metadata,
            "performance": training_session.performance,
            "instructions": {
                "note": "Only models from completed training sessions are stored in MinIO",
                "status_must_be": "completed"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model artifact info: {str(e)}"
        )


# @h2o_router.get("/artifacts/session/{session_id}")
def get_session_artifact_info(session_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Get information about a session artifact in MinIO.

    Parameters:
    -----------
    session_id : str
        The session ID
    user_id : str
        The user ID (must be the owner of the session)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing session artifact information and metadata from the database
    """
    try:
        training_session, _, _, _ = _get_session_context(db, session_id, user_id)

        if not training_session.session_object_key:
            raise HTTPException(
                status_code=404,
                detail=f"No session artifact found for session {session_id}"
            )

        return {
            "status": "success",
            "session_id": session_id,
            "session_object_key": training_session.session_object_key,
            "training_status": training_session.status,
            "created_at": training_session.created_at.isoformat() if training_session.created_at else None,
            "metadata": training_session.metadata,
            "performance": training_session.performance,
            "instructions": {
                "note": "Session artifacts are saved for all training sessions (success, failure, or error)",
                "includes": ["session_data.json", "leaderboard.json", "details_performance.json", "ml_recommendations.txt", "prompt_suggestion.json"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session artifact info: {str(e)}"
        )


# @h2o_router.post("/artifacts/download-model/{session_id}")
async def download_model_artifact_endpoint(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Download a trained model artifact from MinIO.
    Only available for completed training sessions.

    Parameters:
    -----------
    session_id : str
        The session ID
    user_id : str
        The user ID (must be the owner of the session)

    Returns:
    --------
    FileResponse or Dict[str, Any]
        The model artifact ZIP file or error information
    """
    try:
        training_session, _, _, user_uuid = _get_session_context(db, session_id, user_id)

        if not training_session.model_object_key:
            raise HTTPException(
                status_code=404,
                detail=f"No model artifact found for session {session_id}. "
                       f"Models are only saved for successfully completed training sessions."
            )

        if training_session.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Model cannot be downloaded. Training session status is '{training_session.status}'. "
                       f"Only 'completed' sessions have model artifacts."
            )

        # Download model to temporary location
        temp_dir = Path("temp_downloads") / str(user_uuid) / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        download_model_artifact(training_session.model_object_key, temp_dir)

        # Find the extracted model directory
        model_files = list(temp_dir.glob("*"))
        if not model_files:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract model artifact"
            )

        model_dir = model_files[0] if model_files[0].is_dir() else temp_dir

        return {
            "status": "success",
            "message": "Model artifact downloaded successfully",
            "session_id": session_id,
            "model_artifact_key": training_session.model_object_key,
            "download_location": str(model_dir),
            "instructions": {
                "note": "Model is extracted and ready to use with H2O",
                "usage": "Use h2o.load_model() to load the downloaded model"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading model artifact: {str(e)}"
        )


# @h2o_router.post("/artifacts/download-session/{session_id}")
async def download_session_artifact_endpoint(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Download a session artifact from MinIO.
    Available for all training sessions (success, failure, or error).

    Parameters:
    -----------
    session_id : str
        The session ID
    user_id : str
        The user ID (must be the owner of the session)

    Returns:
    --------
    Dict[str, Any]
        Status information about the download
    """
    try:
        training_session, _, _, user_uuid = _get_session_context(db, session_id, user_id)

        if not training_session.session_object_key:
            raise HTTPException(
                status_code=404,
                detail=f"No session artifact found for session {session_id}"
            )

        # Download session to temporary location
        temp_dir = Path("temp_downloads") / str(user_uuid) / session_id / "session"
        temp_dir.mkdir(parents=True, exist_ok=True)

        download_session_artifact(training_session.session_object_key, temp_dir)

        return {
            "status": "success",
            "message": "Session artifact downloaded successfully",
            "session_id": session_id,
            "session_artifact_key": training_session.session_object_key,
            "training_status": training_session.status,
            "download_location": str(temp_dir),
            "contains": [
                "session_data.json - Complete session information",
                "leaderboard.json - Model leaderboard",
                "details_performance.json - Detailed performance metrics",
                "ml_recommendations.txt - ML recommendations",
                "prompt_suggestion.json - AI-generated test prompts"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading session artifact: {str(e)}"
        )


# @h2o_router.get("/artifacts/minio-status")
def check_minio_status(user_id: str, db: Session = Depends(get_db)):
    """
    Check MinIO connection and list user's artifact storage status.

    Parameters:
    -----------
    user_id : str
        The user ID (UUID format)

    Returns:
    --------
    Dict[str, Any]
        Status information about MinIO and user's artifacts
    """
    try:
        _, user_uuid = _get_user_or_404(db, user_id)

        artifacts = list_user_artifacts(user_uuid)

        return {
            "status": "success",
            "minio_connection": "healthy",
            "user_id": str(user_uuid),
            "storage_summary": {
                "total_models_stored": len(artifacts["models"]),
                "total_sessions_stored": len(artifacts["sessions"]),
                "model_storage_size_mb": sum(m["size"] for m in artifacts["models"]) / (1024 * 1024),
                "session_storage_size_mb": sum(s["size"] for s in artifacts["sessions"]) / (1024 * 1024)
            },
            "notes": {
                "models_stored_only_on_success": "Models are only saved in MinIO for completed training sessions",
                "sessions_stored_all_outcomes": "Session data is saved for all outcomes (success, failure, error)",
                "user_isolation": "Users can only access their own data (user_id isolation in storage paths)"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking MinIO status: {str(e)}"
        )
