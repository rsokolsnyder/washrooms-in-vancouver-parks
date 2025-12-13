# download_csv.py
# author: Harry Yau
# date 2025-12-02
import pandas as pd
import os

def download_and_save_csv(url: str, write_to: str, filename: str = "parks.csv"):
    """Download a CSV file from a URL and save it to local directory."""
    try:
        df = pd.read_csv(url, sep=";")
    except Exception as e:
        raise Exception("File appears to be in the incorrect format!")
    
    output_path = os.path.join(write_to, filename)
    df.to_csv(output_path, index=False)
    return output_path