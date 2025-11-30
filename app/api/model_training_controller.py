from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
import uuid
from datetime import datetime
import json
import pandas as pd
import h2o
from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline, evaluate_model_performance_from_path, predict_with_model
from app.helper.utils import save_session_data_to_files, h2o_sessions, get_model_path_from_session_data, SESSION_DATA_DIR
from typing import Dict, Any, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.services.artifact_store import (
    upload_model_artifact,
    upload_session_artifact,
    download_session_artifact,
    download_model_artifact,
    list_user_artifacts,
    delete_artifact,
)

h2o_router = APIRouter(tags=["model-training"])


def _get_user_or_404(db: Session, user_id: str):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    user = crud.get_user(db, user_uuid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user, user_uuid


def _get_training_session_record(db: Session, session_id: str):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    training_session = crud.get_training_session(db, session_uuid)
    if not training_session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return training_session, session_uuid


def _get_session_with_access(db: Session, session_id: str, user_id: str):
    _, user_uuid = _get_user_or_404(db, user_id)
    training_session, session_uuid = _get_training_session_record(db, session_id)
    if training_session.user_id != user_uuid:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    return training_session, session_uuid, user_uuid


def _ensure_session_directory(training_session, session_id: str):
    session_dir = SESSION_DATA_DIR / session_id
    primary_file = session_dir / "session_data.json"
    if not primary_file.exists():
        if training_session.session_object_key:
            download_session_artifact(training_session.session_object_key, session_dir)
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No session artifacts available for {session_id}",
            )
    return session_dir


def _get_session_context(db: Session, session_id: str, user_id: str):
    training_session, session_uuid, user_uuid = _get_session_with_access(db, session_id, user_id)
    session_dir = _ensure_session_directory(training_session, session_id)
    return training_session, session_dir, session_uuid, user_uuid

@h2o_router.get("/run-h2o-ml-pipeline")
def run_h2o_ml_pipeline_endpoint(
    request: Request,
    data_path: str,
    target_variable: str,
    max_runtime_secs: int,
    model_name: str = "gpt-oss:20b",
    user_id: str = "",
    db: Session = Depends(get_db),
):
    """
    Run H2O ML pipeline with real-time execution logs.
    This endpoint streams the execution progress and results.
    """
    print("🚀 Starting H2O ML Pipeline...")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    _, user_uuid = _get_user_or_404(db, user_id)

    # Generate session ID for this training run
    session_uuid = uuid.uuid4()
    session_id = str(session_uuid)

    crud.create_training_session(
        db,
        session_id=session_uuid,
        user_id=user_uuid,
        status="running",
        metadata={
            "data_path": data_path,
            "target_variable": target_variable,
            "max_runtime_secs": max_runtime_secs,
            "model_name": model_name,
        },
    )

    model_object_key = None
    session_object_key = None
    
    def generate():
        nonlocal model_object_key, session_object_key
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

                    # Store model performance to local files
                    model_performance = evaluate_model_performance_from_path(results['model_path'])
                    if model_performance:
                        yield f"LOG: 📊 Model performance saved to local files:\n"
                        yield f"LOG:   - {model_performance}\n"
                        # Save detailed performance JSON for later retrieval
                        try:
                            session_dir = Path("session_data") / session_id
                            session_dir.mkdir(parents=True, exist_ok=True)
                            details_path = session_dir / "details_performance.json"
                            # Prefer structured JSON if available
                            details = getattr(model_performance, "_metric_json", None)
                            if details is None:
                                # Fallback to string representation
                                with open(details_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
                                    f.write(str(model_performance))
                            else:
                                with open(details_path, "w", encoding="utf-8") as f:
                                    json.dump(details, f, indent=2, default=str)
                            yield f"LOG:   - Detailed performance saved\n"
                        except Exception as e:
                            yield f"LOG: ⚠️ Could not save detailed performance: {e}\n"
                
                # Final success message
                yield "LOG: 🎉 H2O ML Pipeline completed successfully!\n"
                
                # Extract performance metrics (handle both classification and regression)
                performance_data = None
                if results['performance']:
                    try:
                        # Try classification metrics
                        performance_data = {
                            "auc": float(results['performance'].auc()),
                            "logloss": float(results['performance'].logloss())
                        }
                    except:
                        # Try regression metrics
                        try:
                            performance_data = {
                                "rmse": float(results['performance'].rmse()),
                                "mse": float(results['performance'].mse()),
                                "mae": float(results['performance'].mae()) if hasattr(results['performance'], 'mae') else None
                            }
                        except Exception as e:
                            yield f"LOG: ⚠️ Could not extract performance for storage: {e}\n"
                            performance_data = {"error": str(e)}
                
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
                    "performance": performance_data,
                    "ml_recommendations": ml_recommendations,
                    "user_id": str(user_uuid),
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)
                if saved_files:
                    yield f"LOG: 💾 Session data saved to local files:\n"
                    for file_type, file_path in saved_files.items():
                        yield f"LOG:   - {file_type}: {file_path}\n"

                try:
                    model_object_key = upload_model_artifact(results['model_path'], session_id, user_uuid)
                    session_object_key = upload_session_artifact(SESSION_DATA_DIR / session_id, session_id, user_uuid)
                except Exception as artifact_error:
                    yield f"LOG: ⚠️ Failed to upload artifacts to MinIO: {artifact_error}\n"

                crud.update_training_session(
                    db,
                    session_id=session_uuid,
                    status="completed",
                    model_object_key=model_object_key,
                    session_object_key=session_object_key,
                    performance=performance_data,
                    metadata=session_data,
                )
                
                # Return final results as JSON
                final_results = {
                    "status": "completed",
                    "message": "H2O ML Pipeline completed successfully",
                    "session_id": session_id,
                    "model_path": results['model_path'],
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": performance_data,
                    "saved_files": saved_files,
                    "ml_recommendations_available": ml_recommendations is not None,
                    "artifacts": {
                        "model_object_key": model_object_key,
                        "session_object_key": session_object_key,
                    },
                }
                
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                yield "STATUS: COMPLETED\n"  # Explicit completion marker for frontend
                
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
                    "error": results['error'],
                    "user_id": str(user_uuid),
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save failed session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)

                crud.update_training_session(
                    db,
                    session_id=session_uuid,
                    status="failed",
                    metadata=session_data,
                )
                
                final_results = {
                    "status": "failed",
                    "error": results['error'],
                    "message": "H2O ML Pipeline failed",
                    "session_id": session_id,
                    "artifacts": {
                        "model_object_key": model_object_key,
                        "session_object_key": session_object_key,
                    },
                }
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                yield "STATUS: FAILED\n"  # Explicit failure marker for frontend
            
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
                "error": str(e),
                "user_id": str(user_uuid),
            }
            
            # Store in global session storage
            h2o_sessions[session_id] = session_data
            
            # Save error session data to local files
            saved_files = save_session_data_to_files(session_id, session_data)

            crud.update_training_session(
                db,
                session_id=session_uuid,
                status="error",
                metadata=session_data,
            )
            
            final_results = {
                "status": "error",
                "error": str(e),
                "message": "H2O ML Pipeline encountered an error",
                "session_id": session_id,
                "artifacts": {
                    "model_object_key": model_object_key,
                    "session_object_key": session_object_key,
                },
            }
            yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
            yield "STATUS: ERROR\n"  # Explicit error marker for frontend
    
    return StreamingResponse(generate(), media_type="text/plain")

@h2o_router.post("/run-h2o-ml-pipeline-advanced")
async def run_h2o_ml_pipeline_advanced_endpoint(
    file: UploadFile = File(..., description="Dataset file (.xlsx, .xls, or .csv)"),
    target_variable: str = Form(..., description="Target column name"),
    user_id: str = Form(..., description="Owner of the training session"),
    db: Session = Depends(get_db),
):
    """
    Run H2O ML pipeline with advanced configuration via POST request.
    Accepts multipart/form-data with file upload (Excel or CSV) and configuration parameters.
    Returns real-time execution logs.
    """
    if not user_id:
        return {"error": "user_id is required", "status": 400}

    _, user_uuid = _get_user_or_404(db, user_id)

    try:
        # Validate file type
        filename = file.filename
        if not filename.endswith(('.xlsx', '.xls', '.csv')):
            return {"error": "Only Excel (.xlsx, .xls) or CSV (.csv) files are supported", "status": 400}

        max_runtime_secs = 300
        
        # Parse exclude columns
        # exclude_columns_list = [col.strip() for col in exclude_columns.split(",") if col.strip()] if exclude_columns else []
        
        # Set default user instructions if not provided (problem type will be auto-detected)
        user_instructions = f"Please train a model to predict '{target_variable}'. Use a max runtime of {max_runtime_secs} seconds."
        
        # Create temporary directory for uploaded file
        from app.helper.utils import DATASETS_DIR
        temp_dir = DATASETS_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        session_uuid = uuid.uuid4()
        session_id = str(session_uuid)
        temp_filename = f"temp_{session_id}_{filename}"
        temp_file_path = temp_dir / temp_filename
        
        # Save uploaded file
        file_content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Read file and convert to CSV if needed
        try:
            # Check if file is CSV or Excel
            if filename.endswith('.csv'):
                # Read CSV directly
                df = pd.read_csv(temp_file_path)
                csv_file_path = temp_file_path
            else:
                # Read Excel and convert to CSV
                df = pd.read_excel(temp_file_path)
                csv_filename = temp_filename.replace('.xlsx', '.csv').replace('.xls', '.csv')
                csv_file_path = temp_dir / csv_filename
                df.to_csv(csv_file_path, index=False)
            
            # Validate target variable exists
            if target_variable not in df.columns:
                return {
                    "error": f"Target variable '{target_variable}' not found in file. Available columns: {', '.join(df.columns)}",
                    "status": 400
                }
            
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}", "status": 400}
        
        # Build config
        config = {
            "data_path": str(csv_file_path),
            "original_filename": filename,
            "target_variable": target_variable,
            "max_runtime_secs": max_runtime_secs,
            "model_name": "gpt-4o-mini",
            "user_instructions": user_instructions,
            "exclude_columns": [],
            "return_predictions": True,
            "return_leaderboard": True,
            "return_performance": True
        }
        
        crud.create_training_session(
            db,
            session_id=session_uuid,
            user_id=user_uuid,
            status="running",
            metadata={
                "original_filename": filename,
                "target_variable": target_variable,
                "max_runtime_secs": max_runtime_secs,
                "model_name": "gpt-4o-mini",
            },
        )

    except Exception as e:
        return {"error": f"Error parsing request: {str(e)}", "status": 400}
    
    print(f"🚀 Starting Advanced H2O ML Pipeline with uploaded file: {filename}")
    
    model_object_key = None
    session_object_key = None

    def generate():
        nonlocal model_object_key, session_object_key
        try:
            # Import the pipeline function
            from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline, shutdown_h2o
            
            yield "LOG: Starting Advanced H2O Machine Learning Pipeline...\n"
            yield f"LOG: Session ID: {session_id}\n"
            yield f"LOG: Uploaded File: {config['original_filename']}\n"
            yield f"LOG: Dataset Shape: {len(df)} rows x {len(df.columns)} columns\n"
            yield f"LOG: Configuration - Target: {config['target_variable']}, Runtime: {config['max_runtime_secs']}s\n"
            
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
                    try:
                        # Try classification metrics first
                        auc = results['performance'].auc()
                        logloss = results['performance'].logloss()
                        yield f"LOG: 📊 Model Performance - AUC: {auc:.4f}, LogLoss: {logloss:.4f}\n"
                    except:
                        # Fall back to regression metrics
                        try:
                            rmse = results['performance'].rmse()
                            mse = results['performance'].mse()
                            yield f"LOG: 📊 Model Performance - RMSE: {rmse:.4f}, MSE: {mse:.4f}\n"
                        except Exception as e:
                            yield f"LOG: ⚠️ Could not extract performance metrics: {e}\n"
                
                # Leaderboard summary
                if results['leaderboard'] is not None:
                    num_models = len(results['leaderboard'])
                    yield f"LOG: 🏆 Generated {num_models} models in leaderboard\n"
                    
                    # Show top 3 models
                    top_models = results['leaderboard'].head(3)
                    yield "LOG: Top 3 Models:\n"
                    for idx, row in top_models.iterrows():
                        model_id = row['model_id'][:50] + "..." if len(row['model_id']) > 50 else row['model_id']
                        # Try to show AUC if available (classification), otherwise show RMSE (regression)
                        try:
                            if 'auc' in row:
                                yield f"LOG:   {idx+1}. {model_id} - AUC: {row['auc']:.4f}\n"
                            elif 'rmse' in row:
                                yield f"LOG:   {idx+1}. {model_id} - RMSE: {row['rmse']:.4f}\n"
                            else:
                                yield f"LOG:   {idx+1}. {model_id}\n"
                        except:
                            yield f"LOG:   {idx+1}. {model_id}\n"
                
                # Model path
                if results['model_path']:
                    yield f"LOG: 💾 Model saved at: {results['model_path']}\n"

                    # Store model performance details to local files
                    try:
                        adv_model_performance = evaluate_model_performance_from_path(results['model_path'])
                        if adv_model_performance:
                            session_dir = Path("session_data") / session_id
                            session_dir.mkdir(parents=True, exist_ok=True)
                            details_path = session_dir / "details_performance.json"
                            details = getattr(adv_model_performance, "_metric_json", None)
                            if details is None:
                                with open(details_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
                                    f.write(str(adv_model_performance))
                            else:
                                with open(details_path, "w", encoding="utf-8") as f:
                                    json.dump(details, f, indent=2, default=str)
                            yield "LOG: 📊 Detailed model performance saved to local files\n"
                    except Exception as e:
                        yield f"LOG: ⚠️ Could not save detailed performance: {e}\n"
                
                # Final success message
                yield "LOG: 🎉 Advanced H2O ML Pipeline completed successfully!\n"
                
                # Extract performance metrics (handle both classification and regression)
                performance_data = None
                if results['performance']:
                    try:
                        # Try classification metrics
                        performance_data = {
                            "auc": float(results['performance'].auc()),
                            "logloss": float(results['performance'].logloss())
                        }
                    except:
                        # Try regression metrics
                        try:
                            performance_data = {
                                "rmse": float(results['performance'].rmse()),
                                "mse": float(results['performance'].mse()),
                                "mae": float(results['performance'].mae()) if hasattr(results['performance'], 'mae') else None
                            }
                        except Exception as e:
                            yield f"LOG: ⚠️ Could not extract performance for storage: {e}\n"
                            performance_data = {"error": str(e)}
                
                # Store session data
                session_data = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "completed",
                    "data_path": config['data_path'],
                    "original_filename": config['original_filename'],
                    "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
                    "target_variable": config['target_variable'],
                    "max_runtime_secs": config['max_runtime_secs'],
                    "model_name": config['model_name'],
                    "user_instructions": config['user_instructions'],
                    "exclude_columns": config['exclude_columns'],
                    "model_path": results['model_path'],
                    "leaderboard": results['leaderboard'].to_dict(orient='records') if results['leaderboard'] is not None else None,
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": performance_data,
                    "ml_recommendations": ml_recommendations,
                    "user_id": str(user_uuid),
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)
                if saved_files:
                    yield f"LOG: 💾 Session data saved to local files:\n"
                    for file_type, file_path in saved_files.items():
                        yield f"LOG:   - {file_type}: {file_path}\n"

                try:
                    model_object_key = upload_model_artifact(results['model_path'], session_id, user_uuid)
                    session_object_key = upload_session_artifact(SESSION_DATA_DIR / session_id, session_id, user_uuid)
                except Exception as artifact_error:
                    yield f"LOG: ⚠️ Failed to upload artifacts to MinIO: {artifact_error}\n"

                crud.update_training_session(
                    db,
                    session_id=session_uuid,
                    status="completed",
                    model_object_key=model_object_key,
                    session_object_key=session_object_key,
                    performance=performance_data,
                    metadata=session_data,
                )
                
                # Return final results as JSON
                final_results = {
                    "status": "completed",
                    "message": "Advanced H2O ML Pipeline completed successfully",
                    "session_id": session_id,
                    "config": config,
                    "model_path": results['model_path'],
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": performance_data,  # Use the extracted performance_data
                    "saved_files": saved_files,
                    "ml_recommendations_available": ml_recommendations is not None,
                    "artifacts": {
                        "model_object_key": model_object_key,
                        "session_object_key": session_object_key,
                    },
                }
                
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                yield "STATUS: COMPLETED\n"  # Explicit completion marker for frontend
                
            else:
                yield f"LOG: ❌ Advanced Pipeline failed: {results['error']}\n"
                
                # Store failed session data
                session_data = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "failed",
                    "data_path": config['data_path'],
                    "original_filename": config.get('original_filename'),
                    "target_variable": config['target_variable'],
                    "max_runtime_secs": config['max_runtime_secs'],
                    "model_name": config['model_name'],
                    "user_instructions": config['user_instructions'],
                    "exclude_columns": config['exclude_columns'],
                    "error": results['error'],
                    "user_id": str(user_uuid),
                }
                
                # Store in global session storage
                h2o_sessions[session_id] = session_data
                
                # Save failed session data to local files
                saved_files = save_session_data_to_files(session_id, session_data)

                crud.update_training_session(
                    db,
                    session_id=session_uuid,
                    status="failed",
                    metadata=session_data,
                )
                
                final_results = {
                    "status": "failed",
                    "error": results['error'],
                    "message": "Advanced H2O ML Pipeline failed",
                    "session_id": session_id,
                    "config": config,
                    "artifacts": {
                        "model_object_key": model_object_key,
                        "session_object_key": session_object_key,
                    },
                }
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                yield "STATUS: FAILED\n"  # Explicit failure marker for frontend
            
            # Cleanup
            try:
                shutdown_h2o()
                yield "LOG: 🧹 H2O cluster shutdown successfully\n"
            except Exception as e:
                yield f"LOG: ⚠️ Warning: Could not shutdown H2O cluster: {e}\n"
            
            # Cleanup temporary files
            try:
                if temp_file_path.exists():
                    temp_file_path.unlink()
                # Only delete CSV file if it's different from the uploaded file
                if csv_file_path != temp_file_path and csv_file_path.exists():
                    csv_file_path.unlink()
                yield "LOG: 🧹 Temporary files cleaned up\n"
            except Exception as e:
                yield f"LOG: ⚠️ Warning: Could not cleanup temporary files: {e}\n"
                
        except Exception as e:
            error_msg = f"🔥 Error in Advanced H2O ML Pipeline: {str(e)}"
            yield f"LOG: {error_msg}\n"
            
            # Store error session data
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "error",
                "data_path": config.get('data_path'),
                "original_filename": config.get('original_filename'),
                "target_variable": config.get('target_variable'),
                "max_runtime_secs": config.get('max_runtime_secs'),
                "model_name": config.get('model_name'),
                "user_instructions": config.get('user_instructions'),
                "exclude_columns": config.get('exclude_columns'),
                "error": str(e),
                "user_id": str(user_uuid),
            }
            
            # Store in global session storage
            h2o_sessions[session_id] = session_data
            
            # Save error session data to local files
            saved_files = save_session_data_to_files(session_id, session_data)

            crud.update_training_session(
                db,
                session_id=session_uuid,
                status="error",
                metadata=session_data,
            )
            
            final_results = {
                "status": "error",
                "error": str(e),
                "message": "Advanced H2O ML Pipeline encountered an error",
                "session_id": session_id,
                "config": config,
                "artifacts": {
                    "model_object_key": model_object_key,
                    "session_object_key": session_object_key,
                },
            }
            yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
            yield "STATUS: ERROR\n"  # Explicit error marker for frontend
    
    return StreamingResponse(generate(), media_type="text/plain")

@h2o_router.get("/h2o-leaderboard/{session_id}")
def get_h2o_leaderboard(session_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Get the model leaderboard for a specific H2O training session from local files.
    
    Parameters:
    -----------
    session_id : str
        The session ID from a previous H2O ML pipeline run
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the leaderboard data
    """
    try:
        training_session, session_dir, _, _ = _get_session_context(db, session_id, user_id)
        leaderboard_file = session_dir / "leaderboard.json"
        
        if not leaderboard_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Leaderboard not found for session {session_id}"
            )
        
        # Load leaderboard data
        with open(leaderboard_file, 'r') as f:
            leaderboard_data = json.load(f)
        
        # Optionally load session metadata
        session_metadata = {}
        session_file = session_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                session_metadata = {
                    "session_id": session_info.get('session_id'),
                    "created_at": session_info.get('created_at'),
                    "data_path": session_info.get('data_path'),
                    "target_variable": session_info.get('target_variable'),
                    "max_runtime_secs": session_info.get('max_runtime_secs'),
                    "model_path": session_info.get('model_path')
                }
        
        return {
            "session_id": session_id,
            "status": "success",
            "leaderboard": leaderboard_data,
            **session_metadata,
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
            detail=f"Error retrieving leaderboard for session {session_id}: {str(e)}"
        )

@h2o_router.get("/model-performance/{session_id}")
def get_model_performance(session_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Get detailed model performance metrics for a specific H2O training session from local files.
    
    Parameters:
    -----------
    session_id : str
        The session ID from a previous H2O ML pipeline run
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing detailed performance metrics including:
        - Model metadata (algorithm, model category)
        - Performance metrics (MSE, RMSE, R2, AUC, LogLoss, etc.)
        - Confusion matrix
        - Thresholds and metric scores
        - Gains/Lift table
        - Residual and null deviance
    """
    try:
        training_session, session_dir, _, _ = _get_session_context(db, session_id, user_id)
        performance_file = session_dir / "details_performance.json"
        
        if not performance_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Performance details not found for session {session_id}"
            )
        
        # Load performance data
        with open(performance_file, 'r') as f:
            performance_data = json.load(f)
        
        # Extract key metrics for easy access
        key_metrics = {
            "model_name": performance_data.get("model", {}).get("name"),
            "model_category": performance_data.get("model_category"),
            "algorithm": performance_data.get("__meta", {}).get("schema_name"),
            "MSE": performance_data.get("MSE"),
            "RMSE": performance_data.get("RMSE"),
            "R2": performance_data.get("r2"),
            "AUC": performance_data.get("AUC"),
            "PR_AUC": performance_data.get("pr_auc"),
            "LogLoss": performance_data.get("logloss"),
            "Gini": performance_data.get("Gini"),
            "AIC": performance_data.get("AIC"),
            "mean_per_class_error": performance_data.get("mean_per_class_error"),
            "nobs": performance_data.get("nobs")
        }
        
        # Optionally load session metadata
        session_metadata = {}
        session_file = session_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                session_metadata = {
                    "session_id": session_info.get('session_id'),
                    "created_at": session_info.get('created_at'),
                    "target_variable": session_info.get('target_variable'),
                    "data_path": session_info.get('data_path')
                }
        
        return {
            "session_id": session_id,
            "status": "success",
            "key_metrics": key_metrics,
            "full_performance_details": performance_data,
            **session_metadata,
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
            detail=f"Error retrieving performance details for session {session_id}: {str(e)}"
        )

@h2o_router.get("/h2o-ml-recommendations/{session_id}")
def get_h2o_ml_recommendations(session_id: str, user_id: str, db: Session = Depends(get_db)):
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
        training_session, session_dir, _, user_uuid = _get_session_context(db, session_id, user_id)
        
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
                    "model_name": session_data.get('model_name'),
                    "artifacts": {
                        "model_object_key": training_session.model_object_key,
                        "session_object_key": training_session.session_object_key,
                    },
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
            
            if session_data.get('ml_recommendations') and session_data.get("user_id") == str(user_uuid):
                return {
                    "session_id": session_id,
                    "status": "success",
                    "source": "memory_session_data",
                    "ml_recommendations": session_data['ml_recommendations'],
                    "recommendations_length": len(session_data['ml_recommendations']),
                    "created_at": session_data.get('created_at'),
                    "data_path": session_data.get('data_path'),
                    "target_variable": session_data.get('target_variable'),
                    "model_name": session_data.get('model_name'),
                    "artifacts": {
                        "model_object_key": training_session.model_object_key,
                        "session_object_key": training_session.session_object_key,
                    },
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

@h2o_router.get("/model-info/{session_id}")
def get_model_info(session_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Get model info for a specific H2O training session from local files.
    """
    try:
        training_session, session_dir, _, _ = _get_session_context(db, session_id, user_id)
        # get details performance
        details_performance_file = session_dir / "details_performance.json"
        if details_performance_file.exists():
            try:
                with open(details_performance_file, 'r') as f:
                    details_performance = json.load(f)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error getting model info for session {session_id}: {str(e)}"
                }
        return {
            "session_id": session_id,   
            "type": details_performance.get('__meta').get('schema_type'),
            "name": details_performance.get('__meta').get('schema_name')
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting model info for session {session_id}: {str(e)}"
        }



class ModelChatBot:
    """
    A chatbot interface for interacting with deployed H2O models through natural language.
    """
    
    def __init__(self, session_id: str, model_name: str = "gpt-4o-mini"):
        self.session_id = session_id
        self.model_path = None
        self.h2o_model = None
        self.session_data = None
        self.llm = ChatOpenAI(model=model_name)
        self.chat_history = []
        self.model_info = {}
        self.details_performance = {}
        
    def initialize(self) -> Dict[str, Any]:
        """Initialize the chatbot with model data."""
        try:
            # Get model path from session data
            self.model_path = get_model_path_from_session_data(self.session_id)
            if not self.model_path:
                return {
                    "status": "error",
                    "message": "Model path not found for this session"
                }
            
            # Load session data for context
            session_dir = Path("session_data") / self.session_id
            session_file = session_dir / "session_data.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    self.session_data = json.load(f)
            
            # Load details performance
            details_performance_file = session_dir / "details_performance.json"
            if details_performance_file.exists():
                with open(details_performance_file, 'r') as f:
                    self.details_performance = json.load(f)
            
            # Initialize H2O and load model
            try:
                h2o.init()
            except Exception:
                pass  # H2O might already be initialized
            
            self.h2o_model = h2o.load_model(self.model_path)
            
            # Get model information
            self.model_info = {
                "model_id": self.h2o_model.model_id,
                "algorithm": self.details_performance.get('__meta').get('schema_name'),
                "model_category": self.details_performance.get('__meta').get('schema_type'),
                "target_variable": self.session_data.get('target_variable', 'Unknown') if self.session_data else 'Unknown',
                "data_path": self.session_data.get('data_path', 'Unknown') if self.session_data else 'Unknown'
            }
            
            return {
                "status": "success",
                "message": "Model chatbot initialized successfully",
                "model_info": self.model_info
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to initialize model chatbot: {str(e)}"
            }
    
    def parse_prediction_request(self, user_input: str) -> Dict[str, Any]:
        """Parse user input to extract prediction request and data."""
        try:
            # Use LLM to understand the user's intent and extract data
            system_prompt = f"""
                You are an AI assistant helping users interact with a machine learning model.

                Model Information:
                - Model ID: {self.model_info.get('model_id', 'Unknown')}
                - Algorithm: {self.details_performance.get('__meta').get('schema_name')}
                - Target Variable: {self.model_info.get('target_variable', 'Unknown')}
                - Data Source: {self.model_info.get('data_path', 'Unknown')}

                Your task is to:
                1. Determine if the user wants to make a prediction
                2. Extract any data values they provided
                3. Ask for missing required information
                4. Check if the question is related to this specific model
                5. Format the response as JSON

                IMPORTANT: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON.

                Respond with JSON in this exact format:
                {{
                    "intent": "prediction" | "question" | "help" | "unrelated",
                    "has_data": true/false,
                    "data": {{"column_name": "value", ...}},
                    "missing_info": ["list", "of", "missing", "columns"],
                    "response": "your response to the user"
                }}

                If the user asks about topics unrelated to this specific model (like general AI, other models, weather, news, etc.), set intent to "unrelated" and provide a polite message explaining you only help with this specific model.

                Examples:
                - If user says "I want to predict churn for age=35, income=75000, tenure=24", respond:
                {{"intent": "prediction", "has_data": true, "data": {{"age": "35", "income": "75000", "tenure": "24"}}, "missing_info": [], "response": "I'll make a prediction with the provided data."}}

                - If user asks "What does this model do?", respond:
                {{"intent": "question", "has_data": false, "data": {{}}, "missing_info": [], "response": "This model predicts {self.model_info.get('target_variable', 'target')} using {self.details_performance.get('__meta').get('schema_name')}."}}

                - If user asks "What's the weather today?" or "Tell me about ChatGPT", respond:
                {{"intent": "unrelated", "has_data": false, "data": {{}}, "missing_info": [], "response": "I'm a specialized chatbot for the {self.details_performance.get('__meta').get('schema_name')} model that predicts {self.model_info.get('target_variable', 'target values')}. I can only help you with questions related to this specific model, such as making predictions, understanding model performance, or asking about the training data. Please ask me something related to this model."}}
                """
            
            user_prompt = f"User input: {user_input}"
            
            # Combine system prompt and user prompt for Ollama
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.llm.invoke(full_prompt)
            
            # Extract content from LangChain response
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                # Clean the response content to extract JSON
                response_content = response_content.strip()
                
                # Try to find JSON in the response if it's wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    
                    # Validate the parsed response has required fields
                    if all(key in parsed_response for key in ["intent", "has_data", "data", "missing_info", "response"]):
                        return parsed_response
                
                # If no JSON found or invalid structure, try direct parsing
                parsed_response = json.loads(response_content)
                if all(key in parsed_response for key in ["intent", "has_data", "data", "missing_info", "response"]):
                    return parsed_response
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"JSON parsing failed: {e}, response: {response_content[:200]}...")
            
            # Fallback if LLM doesn't return valid JSON - try to extract intent manually
            user_lower = user_input.lower()
            
            # Check if the question is related to the trained model
            model_related_keywords = [
                'model', 'prediction', 'predict', 'forecast', 'estimate', 'accuracy', 'performance',
                'algorithm', 'training', 'data', 'feature', 'target', 'churn', 'classification',
                'regression', 'machine learning', 'ml', 'ai', 'artificial intelligence'
            ]
            
            # Add target variable and algorithm to keywords
            target_var = self.model_info.get('target_variable', '').lower()
            algorithm = self.details_performance.get('__meta').get('schema_name').lower()
            if target_var:
                model_related_keywords.append(target_var)
            if algorithm:
                model_related_keywords.append(algorithm)
            
            # Check if the question is related to the model
            is_model_related = any(keyword in user_lower for keyword in model_related_keywords)
            
            if not is_model_related:
                # Question is unrelated to the trained model
                return {
                    "intent": "unrelated",
                    "has_data": False,
                    "data": {},
                    "missing_info": [],
                    "response": f"I'm a specialized chatbot for the {self.details_performance.get('__meta').get('schema_name')} model that predicts {self.model_info.get('target_variable', 'target values')}. I can only help you with questions related to this specific model. Please ask me something related to this model."
                }
            elif any(word in user_lower for word in ['predict', 'prediction', 'forecast', 'estimate']):
                # Try to extract data from the input
                data = {}
                import re
                
                # Look for key=value patterns (improved regex to handle more complex values)
                patterns = re.findall(r'(\w+)\s*=\s*([^,]+?)(?=\s*,\s*\w+\s*=|\s*and\s*$|$)', user_input)
                for key, value in patterns:
                    # Clean up the value (remove trailing words like "and", "months", etc.)
                    clean_value = value.strip().rstrip(' and').rstrip(' months').rstrip(' month')
                    data[key] = clean_value
                
                return {
                    "intent": "prediction",
                    "has_data": len(data) > 0,
                    "data": data,
                    "missing_info": [],
                    "response": f"I'll help you make a prediction. {'I found some data to use.' if data else 'Please provide the data values for prediction.'}"
                }
            else:
                return {
                    "intent": "question",
                    "has_data": False,
                    "data": {},
                    "missing_info": [],
                    "response": response_content if response_content else "I'm here to help you with your machine learning model. You can ask questions or request predictions."
                }
                
        except Exception as e:
            return {
                "intent": "error",
                "has_data": False,
                "data": {},
                "missing_info": [],
                "response": f"Error processing request: {str(e)}"
            }
    
    def make_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the loaded model."""
        try:
            # Convert data to DataFrame
            df = pd.DataFrame([data])
            
            # Get the original training data to understand column types
            try:
                # Load the original dataset to get proper data types
                data_path = self.session_data.get('data_path', 'datasets/churn_data.csv')
                if data_path and data_path.endswith('.csv'):
                    original_df = pd.read_csv(data_path)
                    
                    # Convert columns to match original data types
                    for col in df.columns:
                        if col in original_df.columns:
                            if pd.api.types.is_numeric_dtype(original_df[col]):
                                # Convert to numeric
                                try:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                except:
                                    pass
                            elif pd.api.types.is_categorical_dtype(original_df[col]) or original_df[col].dtype == 'object':
                                # Keep as string/categorical
                                df[col] = df[col].astype(str)
            except Exception as e:
                print(f"Warning: Could not load original dataset for type conversion: {e}")
            
            # Special handling for common numeric fields that might be passed as strings
            numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
            for field in numeric_fields:
                if field in df.columns:
                    try:
                        df[field] = pd.to_numeric(df[field], errors='coerce')
                    except:
                        pass
            
            # Debug: Print data types and sample data
            print(f"DataFrame dtypes: {df.dtypes.astype(str).to_dict()}")
            print(f"DataFrame sample: {df.head().to_dict()}")
            
            # Make prediction
            prediction_result = predict_with_model(self.h2o_model, df, verbose=True)
            
            if prediction_result is not None:
                # Convert prediction to readable format
                prediction_dict = prediction_result.to_dict(orient='records')[0]
                
                # Ensure all values are JSON serializable
                def make_json_safe(obj):
                    if isinstance(obj, dict):
                        return {k: make_json_safe(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_json_safe(item) for item in obj]
                    elif hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    elif hasattr(obj, 'tolist'):  # numpy array
                        return obj.tolist()
                    else:
                        return obj
                
                return {
                    "status": "success",
                    "prediction": make_json_safe(prediction_dict),
                    "input_data": data,
                    "processed_data": make_json_safe(df.to_dict(orient='records')[0]),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to make prediction - no result returned"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Prediction error: {str(e)}"
            }
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Main chat interface."""
        try:
            # Add user input to chat history
            self.chat_history.append({"role": "user", "content": user_input})
            
            # Parse the user's request
            parsed_request = self.parse_prediction_request(user_input)
            
            response = {
                "session_id": self.session_id,
                "user_input": user_input,
                "parsed_request": parsed_request,
                "timestamp": datetime.now().isoformat()
            }
            
            if parsed_request["intent"] == "prediction" and parsed_request["has_data"]:
                # User wants to make a prediction and has provided data
                prediction_result = self.make_prediction(parsed_request["data"])
                response["prediction_result"] = prediction_result
                
                if prediction_result["status"] == "success":
                    # Generate a natural language response about the prediction
                    pred_summary = self._generate_prediction_summary(
                        parsed_request["data"], 
                        prediction_result["prediction"]
                    )
                    response["bot_response"] = pred_summary
                else:
                    response["bot_response"] = prediction_result["message"]
                    
            elif parsed_request["intent"] == "unrelated":
                # User asked something unrelated to the model
                response["bot_response"] = parsed_request["response"]
                response["intent_type"] = "unrelated_question"
                    
            else:
                # User is asking questions or needs help related to the model
                response["bot_response"] = parsed_request["response"]
            
            # Add bot response to chat history
            self.chat_history.append({"role": "assistant", "content": response["bot_response"]})
            
            return response
            
        except Exception as e:
            error_response = {
                "session_id": self.session_id,
                "user_input": user_input,
                "status": "error",
                "bot_response": f"I encountered an error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            return error_response
    
    def _generate_prediction_summary(self, input_data: Dict, prediction: Dict) -> str:
        """Generate a natural language summary of the prediction."""
        try:
            target_var = self.model_info.get('target_variable', 'target')
            
            # Find the main prediction column
            pred_col = None
            for col in prediction.keys():
                if 'predict' in col.lower() or col == target_var:
                    pred_col = col
                    break
            
            if pred_col:
                pred_value = prediction[pred_col]
                
                summary = f"Based on your input data:\n"
                for key, value in input_data.items():
                    summary += f"- {key}: {value}\n"
                
                summary += f"\nThe model predicts: **{pred_value}** for {target_var}\n"
                
                # Add confidence/probability if available
                prob_cols = [col for col in prediction.keys() if 'p0' in col or 'p1' in col or 'prob' in col.lower()]
                if prob_cols:
                    summary += f"\nPrediction probabilities:\n"
                    for col in prob_cols:
                        summary += f"- {col}: {prediction[col]:.4f}\n"
                
                return summary
            else:
                return f"Prediction completed: {prediction}"
                
        except Exception as e:
            return f"Prediction result: {prediction}"

@h2o_router.post("/model-chat/{session_id}")
async def model_chat_endpoint(session_id: str, request: Request, user_id: str, db: Session = Depends(get_db)):
    """
    Chat with a deployed H2O model using natural language.
    """
    try:
        if not user_id:
            return {"error": "user_id is required", "status": 400}

        _get_session_context(db, session_id, user_id)

        data = await request.json()
        user_input = data.get("message", "")
        
        if not user_input:
            return {
                "error": "No message provided",
                "status": 400
            }
        
        print(f"User input: {user_input}")
        # Initialize or get existing chatbot
        chatbot = ModelChatBot(session_id)
        init_result = chatbot.initialize()
        
        if init_result["status"] != "success":
            return {
                "session_id": session_id,
                "status": "error",
                "message": init_result["message"]
            }
        
        # Process the chat message
        chat_response = chatbot.chat(user_input)

        print(f"Chat response: {chat_response}")
        
        return chat_response
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in request body", "status": 400}
    except Exception as e:
        return {
            "session_id": session_id,
            "status": "error",
            "message": f"Chat error: {str(e)}"
        }

@h2o_router.get("/model-deployment/{session_id}")
def model_deployment(session_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Deploy a model for a specific H2O training session and provide chat interface information.
    Includes 5 recommendation prompts for testing the model.
    """
    try:
        if not user_id:
            return {"error": "user_id is required", "status": 400}

        training_session, _, _, _ = _get_session_context(db, session_id, user_id)

        # Initialize chatbot to get model information
        chatbot = ModelChatBot(session_id)
        init_result = chatbot.initialize()
        
        if init_result["status"] != "success":
            return {
                "session_id": session_id,
                "status": "not_found",
                "source": "local_files",
                "message": init_result["message"]
            }
        
        # Get model performance
        model_performance = evaluate_model_performance_from_path(chatbot.model_path)
        
        # Generate 5 recommendation prompts based on model info
        target_variable = chatbot.model_info.get('target_variable', 'target')
        algorithm = chatbot.model_info.get('algorithm', 'Unknown')
        data_path = chatbot.model_info.get('data_path', 'Unknown')
        
        # Try to get feature names from the model for more specific prompts
        feature_names = []
        try:
            if hasattr(chatbot.h2o_model, 'names'):
                feature_names = [name for name in chatbot.h2o_model.names if name != target_variable]
            elif hasattr(chatbot.h2o_model, 'feature_names'):
                feature_names = [name for name in chatbot.h2o_model.feature_names if name != target_variable]
        except:
            # Fallback to common feature names based on data path
            if 'churn' in data_path.lower():
                feature_names = ['age', 'income', 'tenure', 'contract_type', 'monthly_charges']
            else:
                feature_names = ['feature1', 'feature2', 'feature3']
        
        # Detect use case from target variable and data path
        def detect_use_case(target_var, data_path):
            target_lower = target_var.lower()
            path_lower = data_path.lower()
            
            if any(word in target_lower for word in ['churn', 'retention', 'customer']):
                return 'customer_churn'
            elif any(word in target_lower for word in ['price', 'cost', 'value', 'amount']):
                return 'price_prediction'
            elif any(word in target_lower for word in ['weather', 'temperature', 'rain', 'humidity']):
                return 'weather_prediction'
            elif any(word in target_lower for word in ['sales', 'revenue', 'profit']):
                return 'sales_prediction'
            elif any(word in target_lower for word in ['fraud', 'anomaly', 'detection']):
                return 'fraud_detection'
            elif any(word in target_lower for word in ['sentiment', 'review', 'rating']):
                return 'sentiment_analysis'
            elif any(word in target_lower for word in ['stock', 'market', 'trading']):
                return 'financial_prediction'
            elif any(word in target_lower for word in ['health', 'medical', 'diagnosis']):
                return 'healthcare_prediction'
            elif any(word in path_lower for word in ['churn', 'customer']):
                return 'customer_churn'
            elif any(word in path_lower for word in ['weather', 'climate']):
                return 'weather_prediction'
            elif any(word in path_lower for word in ['price', 'cost']):
                return 'price_prediction'
            else:
                return 'general_prediction'
        
        use_case = detect_use_case(target_variable, data_path)
        
        # Generate AI-powered prompts based on actual dataset
        def generate_ai_prompts(target_var, algorithm, features, data_path, use_case):
            try:
                # Initialize OpenAI for AI prompt generation
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini")
                
                # Load and analyze the dataset
                dataset_info = analyze_dataset(data_path, target_var, features)
                
                # Create AI prompt generation request
                ai_prompt_request = f"""
You are an AI assistant helping to generate 5 test prompts for a machine learning model.

Model Information:
- Target Variable: {target_var}
- Algorithm: {algorithm}
- Use Case: {use_case}
- Dataset Path: {data_path}

Dataset Analysis:
- Features: {', '.join(features[:10])}  # Show first 10 features
- Sample Data: {dataset_info.get('sample_data', 'N/A')}
- Data Types: {dataset_info.get('data_types', 'N/A')}
- Value Ranges: {dataset_info.get('value_ranges', 'N/A')}

Please generate 5 diverse, realistic test prompts for this model. Each prompt should:
1. Be specific and actionable for testing the model
2. Use realistic values based on the actual dataset
3. Cover different scenarios (normal, edge cases, etc.)
4. Be written in natural language that a user would actually ask

Format your response as a JSON array with this structure:
[
  {{
    "id": 1,
    "category": "Category Name",
    "prompt": "Actual prompt text with realistic values",
    "description": "Brief description of what this tests"
  }},
  ...
]

Make the prompts specific to the {use_case} use case and use actual feature names and realistic values from the dataset.
"""

                # Generate prompts using AI
                ai_response = llm.invoke(ai_prompt_request)
                
                # Try to parse JSON response
                try:
                    # Clean the response to extract JSON
                    import re
                    json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        ai_prompts = json.loads(json_str)
                        
                        # Validate and clean the prompts
                        validated_prompts = []
                        for i, prompt in enumerate(ai_prompts[:5]):  # Take first 5
                            if isinstance(prompt, dict) and 'prompt' in prompt:
                                validated_prompts.append({
                                    "id": i + 1,
                                    "category": prompt.get('category', f'Test {i+1}'),
                                    "prompt": prompt.get('prompt', ''),
                                    "description": prompt.get('description', f'Test prompt {i+1}')
                                })
                        
                        if len(validated_prompts) >= 3:  # Need at least 3 valid prompts
                            return validated_prompts
                
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"AI response parsing failed: {e}")
                
                # Fallback to simple AI-generated prompts if JSON parsing fails
                fallback_prompt = f"""
Generate 5 test prompts for a {algorithm} model that predicts {target_var}.
Use case: {use_case}
Features: {', '.join(features[:5])}

Return 5 different prompts, each on a new line, starting with "1.", "2.", etc.
Make them realistic and specific to the {use_case} domain.
"""
                
                fallback_response = llm.invoke(fallback_prompt)
                fallback_prompts = []
                
                # Parse line-by-line response
                lines = fallback_response.strip().split('\n')
                for i, line in enumerate(lines[:5]):
                    if line.strip() and any(line.strip().startswith(str(j)) for j in range(1, 6)):
                        # Extract prompt text after number
                        prompt_text = re.sub(r'^\d+\.\s*', '', line.strip())
                        if prompt_text:
                            fallback_prompts.append({
                                "id": i + 1,
                                "category": f"AI Generated {i+1}",
                                "prompt": prompt_text,
                                "description": f"AI-generated test prompt {i+1}"
                            })
                
                if fallback_prompts:
                    return fallback_prompts
                    
            except Exception as e:
                print(f"AI prompt generation failed: {e}")
            
            # Final fallback to basic prompts
            return [
                {
                    "id": 1,
                    "category": "Basic Prediction",
                    "prompt": f"Predict {target_var} with sample data",
                    "description": "Basic prediction test"
                },
                {
                    "id": 2,
                    "category": "Feature Analysis",
                    "prompt": f"Which features are most important for {target_var}?",
                    "description": "Feature importance analysis"
                },
                {
                    "id": 3,
                    "category": "Model Performance",
                    "prompt": f"What is the accuracy of this {algorithm} model?",
                    "description": "Model performance inquiry"
                }
            ]
        
        def analyze_dataset(data_path, target_var, features):
            """Analyze the dataset to extract useful information for prompt generation"""
            try:
                import pandas as pd
                
                # Load dataset
                if data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                else:
                    return {"error": "Unsupported file format"}
                
                # Basic dataset analysis
                dataset_info = {
                    "num_rows": len(df),
                    "num_features": len(features),
                    "data_types": df.dtypes.to_dict(),
                    "sample_data": {},
                    "value_ranges": {}
                }
                
                # Get sample data for key features
                for feature in features[:5]:  # Analyze first 5 features
                    if feature in df.columns:
                        sample_values = df[feature].dropna().head(3).tolist()
                        dataset_info["sample_data"][feature] = sample_values
                        
                        # Get value ranges for numeric features
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            min_val = df[feature].min()
                            max_val = df[feature].max()
                            dataset_info["value_ranges"][feature] = f"{min_val} to {max_val}"
                        else:
                            # For categorical features, get unique values
                            unique_vals = df[feature].value_counts().head(3).index.tolist()
                            dataset_info["value_ranges"][feature] = unique_vals
                
                return dataset_info
                
            except Exception as e:
                return {"error": f"Dataset analysis failed: {e}"}
        
        # Generate AI-powered prompts
        recommendation_prompts = generate_ai_prompts(target_variable, algorithm, feature_names, data_path, use_case)
        
        # Analyze dataset for additional context
        dataset_analysis = analyze_dataset(data_path, target_variable, feature_names)
        
        # Save recommendation prompts to local JSON file
        session_dir = SESSION_DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        prompt_suggestion_file = session_dir / "prompt_suggestion.json"
        
        prompt_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "model_info": chatbot.model_info,
            "detected_use_case": use_case,
            "dataset_analysis": dataset_analysis,
            "ai_generated_prompts": True,
            "recommendation_prompts": recommendation_prompts,
            "usage_instructions": {
                "endpoint": f"POST /model-chat/{session_id}",
                "body_format": {"message": "your question or prediction request"},
                "note": f"AI-generated prompts for {use_case} model testing based on actual dataset",
                "ai_model_used": "qwen2:7b"
            }
        }
        
        with open(prompt_suggestion_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2, default=str)

        try:
            upload_session_artifact(session_dir, session_id, training_session.user_id)
        except Exception as artifact_error:
            print(f"⚠️ Failed to sync prompt suggestions to MinIO: {artifact_error}")
        
        return {
            "session_id": session_id,
            "status": "success",
            "source": "local_files",
            "model_info": chatbot.model_info,
            "detected_use_case": use_case,
            "dataset_analysis": dataset_analysis,
            "ai_generated_prompts": True,
            "model_performance": str(model_performance) if model_performance else None,
            "chat_endpoint": f"/model-chat/{session_id}",
            "recommendation_prompts": recommendation_prompts,
            "prompt_suggestion_file": str(prompt_suggestion_file),
            "usage_instructions": {
                "endpoint": f"POST /model-chat/{session_id}",
                "body_format": {"message": "your question or prediction request"},
                "examples": [
                    "What does this model predict?",
                    "I want to make a prediction with sample data",
                    f"Use the AI-generated {use_case} prompts below for testing"
                ],
                "recommendation_prompts_available": True,
                "use_case": use_case,
                "ai_model_used": "qwen2:7b",
                "prompt_file_saved": str(prompt_suggestion_file)
            }
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "status": "error",
            "message": f"Deployment error: {str(e)}"
        }

@h2o_router.get("/prompt-suggestions/{session_id}")
def get_prompt_suggestions(session_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Get the 5 recommendation prompts for testing a specific model from local JSON file.

    Parameters:
    -----------
    session_id : str
        The session ID from a previous H2O ML pipeline run

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the recommendation prompts and model information
    """
    try:
        training_session, session_dir, _, _ = _get_session_context(db, session_id, user_id)
        prompt_suggestion_file = session_dir / "prompt_suggestion.json"

        if prompt_suggestion_file.exists():
            with open(prompt_suggestion_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)

            return {
                "session_id": session_id,
                "status": "success",
                "source": "local_files",
                "created_at": prompt_data.get('created_at'),
                "model_info": prompt_data.get('model_info'),
                "recommendation_prompts": prompt_data.get('recommendation_prompts', []),
                "usage_instructions": prompt_data.get('usage_instructions'),
                "file_path": str(prompt_suggestion_file)
            }
        else:
            return {
                "session_id": session_id,
                "status": "not_found",
                "source": "local_files",
                "message": "Prompt suggestions file not found. Please run model deployment first."
            }

    except Exception as e:
        return {
            "session_id": session_id,
            "status": "error",
            "message": f"Error retrieving prompt suggestions: {str(e)}"
        }


# ============================================================================
# Model History and MinIO Artifact Management Endpoints
# ============================================================================

@h2o_router.get("/model-history/{user_id}")
def get_model_history(user_id: str, db: Session = Depends(get_db)):
    """
    Get complete model training history for a user with MinIO artifact information.
    This endpoint displays all training sessions owned by the user for the model history page.

    Parameters:
    -----------
    user_id : str
        The user ID (UUID format)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all training sessions with metadata, artifacts, and performance info
    """
    try:
        _, user_uuid = _get_user_or_404(db, user_id)

        # Get all training sessions for this user from database
        training_sessions = crud.get_sessions_for_user(db, user_uuid)

        if not training_sessions:
            return {
                "status": "success",
                "user_id": str(user_uuid),
                "total_sessions": 0,
                "sessions": [],
                "message": "No training sessions found for this user"
            }

        # Build response with session details
        sessions_list = []
        for session in training_sessions:
            session_info = {
                "session_id": str(session.session_id),
                "status": session.status,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "metadata": session.metadata_json,
                "performance": session.performance,
                "artifacts": {
                    "model_object_key": session.model_object_key,
                    "session_object_key": session.session_object_key,
                    "has_model": session.model_object_key is not None,
                    "has_session_data": session.session_object_key is not None
                },
                "model_info": {
                    "target_variable": session.metadata_json.get("target_variable") if session.metadata_json else None,
                    "original_filename": session.metadata_json.get("original_filename") if session.metadata_json else None,
                    "model_path": session.metadata_json.get("model_path") if session.metadata_json else None,
                    "num_models": session.metadata_json.get("num_models") if session.metadata_json else 0,
                }
            }
            sessions_list.append(session_info)

        # Sort by created_at descending (most recent first)
        sessions_list.sort(key=lambda x: x["created_at"] or "", reverse=True)

        # Count sessions by status
        status_counts = {}
        for session in sessions_list:
            status = session["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "status": "success",
            "user_id": str(user_uuid),
            "total_sessions": len(sessions_list),
            "status_counts": status_counts,
            "sessions": sessions_list,
            "note": "Only 'completed' sessions have trained model artifacts. All sessions have session data artifacts."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model history: {str(e)}"
        )


@h2o_router.get("/artifacts/list")
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


@h2o_router.get("/artifacts/model/{session_id}")
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
            "metadata": training_session.metadata_json,
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


@h2o_router.get("/artifacts/session/{session_id}")
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
            "metadata": training_session.metadata_json,
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


@h2o_router.post("/artifacts/download-model/{session_id}")
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
    Dict[str, Any]
        Status information about the download
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


@h2o_router.post("/artifacts/download-session/{session_id}")
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


@h2o_router.get("/artifacts/minio-status")
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