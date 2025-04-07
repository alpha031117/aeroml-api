import re
import shutil
import os

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

