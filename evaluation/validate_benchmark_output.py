
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "test_datasets" / "nyc_311_large.csv"
OUTPUT_FILE = BASE_DIR / "benchmark_outputs" / "nyc_311_cleaned.csv"
CHUNK_SIZE = 50000
MAX_MISMATCHES = 20

def diagnose():
    input_rows = 0
    output_rows = 0
    mismatch_counts = {}
    examples = []

    original_reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False
    )

    cleaned_reader = pd.read_csv(
        OUTPUT_FILE,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False
    )

    chunk_number = 0

    while True:
        original = next(original_reader, None)
        cleaned = next(cleaned_reader, None)

        if original is None and cleaned is None:
            break

        chunk_number += 1

        if original is None or cleaned is None:
            print(f"Chunk {chunk_number}: missing from one file")
            break

        input_rows += len(original)
        output_rows += len(cleaned)

        if list(original.columns) != list(cleaned.columns):
            print("Column names or order differ.")
            print("Input:", list(original.columns))
            print("Output:", list(cleaned.columns))
            return

        if len(original) != len(cleaned):
            print(f"Row count mismatch in chunk {chunk_number}")
            print("Input:", len(original), "Output:", len(cleaned))
            continue

        differences = original.ne(cleaned)

        for column in differences.columns:
            count = int(differences[column].sum())

            if count:
                mismatch_counts[column] = (
                    mismatch_counts.get(column, 0) + count
                )

                if len(examples) < MAX_MISMATCHES:
                    positions = differences.index[differences[column]]

                    for position in positions:
                        if len(examples) >= MAX_MISMATCHES:
                            break

                        examples.append({
                            "chunk": chunk_number,
                            "row_in_chunk": int(position - differences.index[0]),
                            "column": column,
                            "input": repr(original.loc[position, column]),
                            "output": repr(cleaned.loc[position, column])
                        })

    print("\nINTEGRITY DIAGNOSTIC")
    print("=" * 60)
    print("Input rows:", input_rows)
    print("Output rows:", output_rows)

    print("\nMISMATCH COUNTS BY COLUMN")

    if not mismatch_counts:
        print("No value mismatches found.")
    else:
        for column, count in sorted(
            mismatch_counts.items(),
            key=lambda item: item[1],
            reverse=True
        ):
            print(f"{column}: {count}")

    print("\nFIRST MISMATCH EXAMPLES")

    for item in examples:
        print(item)

if __name__ == "__main__":
    diagnose()
