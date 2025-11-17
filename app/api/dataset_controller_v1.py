from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import subprocess
import shutil
import json
import pandas as pd
from typing import Optional
from app.helper.generate_LoRa_dataset import suggest_sources
from app.helper.utils import DATASETS_DIR

dataset_router = APIRouter(tags=["dataset-elicitation"])

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
