from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import subprocess
import json
import asyncio
import sys
import os
from generate_LoRa_dataset import suggest_sources

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

@app.get("/")
def read_root():
    query = "Weather Prediction at Malaysia"
    sources = suggest_sources(query)
    return {"sources": sources}

@app.post("/suggest-sources")
async def suggest_sources_endpoint(request: Request):
    data = await request.json()  # Use await to get the data
    query = data.get("modelInput", "")  # Now you can use get safely
    sources = suggest_sources(query)  # Call the function to get sources
    return {"sources": sources}  # Return the sources in the response

@app.get("/run-stagehand")
def run_stagehand_script(request: Request):
    print("🔥 Starting subprocess...")
    def generate():
        context = None

        # Use full path to `npx.cmd` on Windows
        npm_cmd = "C:\\Users\\alpha\\AppData\\Roaming\\npm\\npm.cmd"  # ✅ Adjust if different
        ts_file = os.path.join("aeroml-browserbase", "index.ts")

        try:
            process = subprocess.Popen(
                [npm_cmd, "run", "stagehand"],  # or ["node", "--loader", ...] if you're running directly
                cwd=os.path.join(os.getcwd()),  # or the correct subfolder
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                encoding="utf-8",         # ✅ Force UTF-8 decoding
                errors="replace"          # ✅ Replace bad chars with � instead of crashing
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