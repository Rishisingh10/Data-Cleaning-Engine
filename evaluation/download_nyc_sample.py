
from pathlib import Path

import requests


OUTPUT_DIR = Path(__file__).resolve().parent / "test_datasets"
OUTPUT_FILE = OUTPUT_DIR / "nyc_311_sample.csv"

API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"


def download_dataset(row_limit=10000):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "$limit": row_limit,
        "$order": "created_date DESC"
    }

    print(f"Downloading {row_limit} NYC 311 records...")

    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    if not response.content.strip():
        raise ValueError("The API returned an empty dataset.")

    OUTPUT_FILE.write_bytes(response.content)

    print("Download successful!")
    print(f"File: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    download_dataset()
