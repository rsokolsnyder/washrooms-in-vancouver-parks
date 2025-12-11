# download_data.py
# author: Harry Yau
# date 2025-12-02

import click
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
    
@click.command()
@click.option('--url', type=str, help="URL of dataset to be download")
@click.option('--write_to', type=str, help="Path for the saving location")
def main(url, write_to):
    """Download data from the website to local directory with the path."""
    download_and_save_csv(url, write_to)
    
if __name__ == '__main__':
    main()
