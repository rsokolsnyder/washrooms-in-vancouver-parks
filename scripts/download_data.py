# download_data.py
# author: Harry Yau
# date 2025-12-02

import click
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from download_csv import download_and_save_csv
    
@click.command()
@click.option('--url', type=str, help="URL of dataset to be download")
@click.option('--write_to', type=str, help="Path for the saving location")
def main(url, write_to):
    """Download data from the website to local directory with the path."""
    download_and_save_csv(url, write_to)
    
if __name__ == '__main__':
    main()
