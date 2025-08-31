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

import json
import asyncio
import sys
import os
import logging
from pathlib import Path
import pandas as pd
from app.helper.utils import list_dataset_name, DATASETS_DIR
from typing import Dict, Any, Optional

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
from datetime import datetime

from app.api.dataset_controller import dataset_router
from app.api.model_training_controller import h2o_router
from app.api.h2o_utils_controller import h2o_utils_router

def get_application() -> FastAPI:
    application = FastAPI(
    title="AeroML API",
    description="FastAPI application for dataset elicitation and H2O ML pipelines",
    version="1.0.0",
    openapi_tags=[
        {"name": "dataset-elicitation", "description": "Discover/prepare datasets & utilities"},
        {"name": "model-training", "description": "Run H2O pipelines, manage sessions & results"},
    ],
    )


    # Event loop policy for compatibility (mostly safe fallback)
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Allow CORS for specific origin
    origins = [
        "http://localhost:3000",  # React frontend
        "http://127.0.0.1:80",   # If you want to allow backend to frontend communication
    ]
        
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(dataset_router, prefix="/api/dataset")
    application.include_router(h2o_router, prefix="/api/model-training")
    application.include_router(h2o_utils_router, prefix="/api/model-training-utils")

    return application


app = get_application()

if __name__ == '__main__':
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True, log_level="debug")




