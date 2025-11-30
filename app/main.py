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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import uvicorn
# from app.api.dataset_controller import dataset_router
from app.api.dataset_controller import dataset_validation_router
from app.api.model_training_controller import h2o_router
from app.api.h2o_utils_controller import h2o_utils_router
from app.api.user_controller import user_router
from app.db.database import init_db

def get_application() -> FastAPI:
    application = FastAPI(
    title="AeroML API",
    description="FastAPI application for dataset elicitation and H2O ML pipelines",
    version="1.0.0",
    openapi_tags=[
        {"name": "dataset-validation", "description": "Validate datasets using OpenAI for training suitability"},
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

    # application.include_router(dataset_router, prefix="/api/dataset")
    application.include_router(dataset_validation_router, prefix="/api/dataset")
    application.include_router(h2o_router, prefix="/api/model-training")
    application.include_router(h2o_utils_router, prefix="/api/model-training-utils")
    application.include_router(user_router, prefix="/api/users")

    return application


app = get_application()


@app.on_event("startup")
def startup_event():
    init_db()

if __name__ == '__main__':
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")




