import re
import shutil
import os
from pathlib import Path
from fastapi import HTTPException
from app.helper.logger import get_logger
import json
import pandas as pd
from typing import Dict, Any


DATASETS_DIR = Path("datasets")

# Global session storage
h2o_sessions: Dict[str, Dict[str, Any]] = {}

# Create directory for storing session data
SESSION_DATA_DIR = Path("session_data")
SESSION_DATA_DIR.mkdir(exist_ok=True)

logger = get_logger()

def await_disconnect(request) -> bool:
    """
    Detect if client has disconnected. For now, just return False,
    or customize with request-specific checks if needed.
    """
    if hasattr(request, "is_disconnected"):
        return request.is_disconnected()
    return False

def extract_urls(text):
    """Extracts unique URLs from a given text, removing markdown-style duplicates."""
    # Regex pattern to find URLs inside Markdown-style links or normal text
    url_pattern = r"https?://[^\s\)\]]+"  
    urls = re.findall(url_pattern, text)

    # Remove duplicates and return clean list
    return list(set(urls))  # `set()` removes duplicates

def convert_size_to_bytes(size_str):
    """Convert dataset size from KB, MB, GB format to bytes."""
    size_mapping = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
    match = re.match(r"([\d.]+)(KB|MB|GB)", size_str)

    if match:
        size_value, unit = match.groups()
        return int(float(size_value) * size_mapping[unit])
    return 0  # Return 0 if size is not recognized

def delete_datasets_directory(directory="datasets"):
    """Deletes the datasets directory after the program completes execution."""
    if os.path.exists(directory):
        print(f"\n🗑️ Deleting directory: {directory}...")
        shutil.rmtree(directory)
        print("✅ Datasets directory deleted successfully.")
    else:
        print("⚠️ Datasets directory does not exist. Nothing to delete.")


def list_dataset_name() -> str:
    """
    Return the name of a CSV file to use by default.
    Strategy: latest modified CSV under DATASETS_DIR.
    Raises 404 if the folder or CSVs don't exist.
    """
    if not DATASETS_DIR.exists():
        raise HTTPException(status_code=404, detail="Datasets directory not found")

    csvs = list(DATASETS_DIR.glob("*.csv"))
    if not csvs:
        raise HTTPException(status_code=404, detail="No CSV files found in datasets directory")

    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0].name  # e.g., "mydata.csv"

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