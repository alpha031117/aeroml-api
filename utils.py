import re
import shutil
import os
from pathlib import Path
from fastapi import HTTPException
import logging

DATASETS_DIR = Path("datasets")

def get_logger(name=__name__):
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)  # Only store INFO and above in file

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Show DEBUG and above in console

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Prevent adding multiple handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

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
