import os
import pandas as pd
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from download_csv import download_and_save_csv

def test_download_and_save_csv_success(tmp_path):
    """Test that CSV is downloaded and saved correctly."""

    # Example public CSV with semicolon separator
    url = "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/parks/exports/csv?lang=en&timezone=America%2FLos_Angeles&use_labels=true&delimiter=%3B"
    output_dir = tmp_path

    output_path = download_and_save_csv(
        url=url,
        write_to=output_dir,
        filename="test.csv"
    )

    # Check file exists
    assert os.path.exists(output_path)

    # Check file is readable
    df = pd.read_csv(output_path)
    assert not df.empty

def test_download_and_save_csv_invalid_url(tmp_path):
    """Test that invalid CSV raises an exception."""

    bad_url = "https://example.com/not_a_csv.txt"

    with pytest.raises(Exception, match="incorrect format"):
        download_and_save_csv(
            url=bad_url,
            write_to=tmp_path
        )