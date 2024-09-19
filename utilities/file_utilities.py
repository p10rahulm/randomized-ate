import os
from datetime import datetime

import os

def get_latest_file(directory, filetype='pth'):
    """
    Retrieves the latest file from the given directory based on the last modified time.
    Filters by the specified file type if provided.

    Args:
    directory (str): The path to the directory to search in.
    filetype (str): The file extension to filter by (e.g., 'pth'). Set to an empty string to include all file types.

    Returns:
    str: The full path of the latest file, or None if no files are found.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return None

    # List all files in the directory, filtering by file type if specified
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and (f.endswith(f'.{filetype}') if filetype else True)
    ]
    
    if not files:
        print(f"No {'.' + filetype if filetype else ''} files found in the directory.")
        return None

    # Find the latest file based on the last modified time
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

