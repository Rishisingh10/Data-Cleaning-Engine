
import time
from pathlib import Path

import requests


OUTPUT_DIR = Path(__file__).resolve().parent / "test_datasets"
OUTPUT_FILE = OUTPUT_DIR / "nyc_311_large.csv"

API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"

TOTAL_RECORDS = 1_000_000
BATCH_SIZE = 50_000

HEADERS = {
    "User-Agent": "DataDoctorAI-Evaluation/1.0"
}


def download_large_dataset():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    downloaded = 0
    session = requests.Session()
    session.headers.update(HEADERS)

    while downloaded < TOTAL_RECORDS:

        limit = min(BATCH_SIZE, TOTAL_RECORDS - downloaded)

        params = {
            "$limit": limit,
            "$offset": downloaded,
            "$order": "unique_key ASC"
        }

        print(
            f"Downloading records "
            f"{downloaded + 1:,} to {downloaded + limit:,}..."
        )

        for attempt in range(4):
            try:
                response = session.get(
                    API_URL,
                    params=params,
                    timeout=(20, 300)
                )

                response.raise_for_status()

                content = response.content

                if not content.strip():
                    print("No more records available.")
                    break

                lines = content.splitlines()

                if not lines:
                    break

                if downloaded == 0:
                    with open(OUTPUT_FILE, "wb") as file:
                        file.write(content)
                else:
                    with open(OUTPUT_FILE, "ab") as file:
                        file.write(b"\n")
                        file.write(b"\n".join(lines[1:]))

                rows_received = max(len(lines) - 1, 0)
                downloaded += rows_received

                print(
                    f"Downloaded: {downloaded:,} records | "
                    f"File size: "
                    f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
                )

                break

            except requests.RequestException as error:
                print(f"Request failed: {error}")

                if attempt == 3:
                    raise

                time.sleep(2 ** attempt)

        else:
            raise RuntimeError("Unable to download the next batch.")

        if downloaded >= TOTAL_RECORDS:
            break

        if rows_received == 0:
            break

        time.sleep(0.5)

    print("\nDOWNLOAD COMPLETED")
    print(f"Records downloaded: {downloaded:,}")
    print(f"File location: {OUTPUT_FILE}")
    print(
        f"Final size: "
        f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )


if __name__ == "__main__":
    download_large_dataset()
