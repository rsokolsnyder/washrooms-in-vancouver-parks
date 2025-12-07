# download_data.py
# author: Harry Yau
# date 2025-12-02

import click
import pandas as pd
import os
@click.command()
@click.option('--url', type=str, help="URL of dataset to be download")
@click.option('--write_to', type=str, help="Path for the saving location")
def main(url, write_to):
    """Download data from the website to local directory with the path."""
    try:
        park = pd.read_csv(url, sep=";")
    except:
        raise Exception("File appears to be in the incorrect format!")
    park.to_csv(os.path.join(write_to, "parks.csv"), index=False)
if __name__ == '__main__':
    main()
