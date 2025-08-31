from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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
from fastapi.responses import FileResponse
import pandas as pd
from utils import list_dataset_name
from fastapi import HTTPException
from utils import DATASETS_DIR
from typing import Optional

# Import the pipeline function
from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline, shutdown_h2o


# Event loop policy for compatibility (mostly safe fallback)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

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

# @app.get("/")
# def read_root():
#     query = "Weather Prediction at Malaysia"
#     sources = suggest_sources(query)
#     return {"sources": sources}

@app.post("/suggest-sources")
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

@app.get("/datasets")
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

@app.get("/model-training")
def model_training(request: Request):
    return {"message": "Model training started"}

@app.get("/h2o-ml-pipeline-status")
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

@app.get("/run-h2o-ml-pipeline")
def run_h2o_ml_pipeline_endpoint(
    request: Request,
    data_path: str = "datasets/churn_data.csv",
    target_variable: str = "Churn",
    max_runtime_secs: int = 30,
    model_name: str = "gpt-4o-mini"
):
    """
    Run H2O ML pipeline with real-time execution logs.
    This endpoint streams the execution progress and results.
    """
    print("🚀 Starting H2O ML Pipeline...")
    
    def generate():
        try:
            yield "LOG: Starting H2O Machine Learning Pipeline...\n"
            
            # Run the pipeline with verbose output
            results = run_h2o_ml_pipeline(
                data_path=data_path,
                config_path="config/credentials.yml",
                target_variable=target_variable,
                user_instructions=f"Please do classification on '{target_variable}'. Use a max runtime of {max_runtime_secs} seconds.",
                model_name=model_name,
                max_runtime_secs=max_runtime_secs,
                log_enabled=True,
                return_model=True,
                return_predictions=True,
                return_leaderboard=True,
                return_performance=True,
                verbose=True
            )
            
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
                
                # Return final results as JSON
                final_results = {
                    "status": "completed",
                    "message": "H2O ML Pipeline completed successfully",
                    "model_path": results['model_path'],
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": {
                        "auc": float(results['performance'].auc()) if results['performance'] else None,
                        "logloss": float(results['performance'].logloss()) if results['performance'] else None
                    } if results['performance'] else None
                }
                
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                
            else:
                yield f"LOG: ❌ Pipeline failed: {results['error']}\n"
                final_results = {
                    "status": "failed",
                    "error": results['error'],
                    "message": "H2O ML Pipeline failed"
                }
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
            
            # Cleanup
            try:
                shutdown_h2o()
                yield "LOG: 🧹 H2O cluster shutdown successfully\n"
            except Exception as e:
                yield f"LOG: ⚠️ Warning: Could not shutdown H2O cluster: {e}\n"
                
        except Exception as e:
            error_msg = f"🔥 Error in H2O ML Pipeline: {str(e)}"
            yield f"LOG: {error_msg}\n"
            final_results = {
                "status": "error",
                "error": str(e),
                "message": "H2O ML Pipeline encountered an error"
            }
            yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/run-h2o-ml-pipeline-advanced")
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
    
    print(f"🚀 Starting Advanced H2O ML Pipeline with config: {config}")
    
    def generate():
        try:
            # Import the pipeline function
            from h2o_machine_learning_agent.h2o_ml_pipeline import run_h2o_ml_pipeline, shutdown_h2o
            
            yield "LOG: Starting Advanced H2O Machine Learning Pipeline...\n"
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
                
                # Return final results as JSON
                final_results = {
                    "status": "completed",
                    "message": "Advanced H2O ML Pipeline completed successfully",
                    "config": config,
                    "model_path": results['model_path'],
                    "num_models": len(results['leaderboard']) if results['leaderboard'] is not None else 0,
                    "performance": {
                        "auc": float(results['performance'].auc()) if results['performance'] else None,
                        "logloss": float(results['performance'].logloss()) if results['performance'] else None
                    } if results['performance'] else None
                }
                
                yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
                
            else:
                yield f"LOG: ❌ Advanced Pipeline failed: {results['error']}\n"
                final_results = {
                    "status": "failed",
                    "error": results['error'],
                    "message": "Advanced H2O ML Pipeline failed",
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
            final_results = {
                "status": "error",
                "error": str(e),
                "message": "Advanced H2O ML Pipeline encountered an error",
                "config": config
            }
            yield f"FINAL_RESULT: {json.dumps(final_results, indent=2)}\n"
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/run-stagehand")
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