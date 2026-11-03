from pathlib import Path
import json

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "test_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TOTAL_BASE_ROWS = 1000

rng = np.random.default_rng(RANDOM_SEED)

data = pd.DataFrame({
    "customer_id": [f"CUST{i:05d}" for i in range(1, TOTAL_BASE_ROWS + 1)],
    "age": rng.integers(18, 70, TOTAL_BASE_ROWS),
    "annual_income": np.round(rng.normal(65000, 12000, TOTAL_BASE_ROWS), 2),
    "city": rng.choice(
        ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad"],
        TOTAL_BASE_ROWS
    ),
    "email": [f"customer{i}@example.com" for i in range(1, TOTAL_BASE_ROWS + 1)],
    "signup_date": pd.date_range("2023-01-01", periods=TOTAL_BASE_ROWS, freq="D").astype(str)
})

ground_truth = {
    "random_seed": RANDOM_SEED,
    "base_rows": TOTAL_BASE_ROWS,
    "injected_issues": {}
}

missing_indices = list(range(0, 50))
data.loc[missing_indices, "annual_income"] = np.nan

ground_truth["injected_issues"]["missing_values"] = {
    "column": "annual_income",
    "affected_rows": missing_indices,
    "expected_count": len(missing_indices)
}

formatting_indices = list(range(50, 90))
formatting_values = [
    " mumbai ",
    "DELHI",
    "bangalore ",
    " PUNE",
    "hyderabad "
]

for index in formatting_indices:
    original_city = data.loc[index, "city"]
    position = ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad"].index(original_city)
    data.loc[index, "city"] = formatting_values[position]

ground_truth["injected_issues"]["inconsistent_categorical_formatting"] = {
    "column": "city",
    "affected_rows": formatting_indices,
    "expected_count": len(formatting_indices)
}

outlier_indices = list(range(90, 100))
data.loc[outlier_indices, "annual_income"] = [
    500000,
    550000,
    600000,
    650000,
    700000,
    750000,
    800000,
    850000,
    900000,
    1000000
]

ground_truth["injected_issues"]["potential_outliers"] = {
    "column": "annual_income",
    "affected_rows": outlier_indices,
    "expected_count": len(outlier_indices)
}

duplicate_source_indices = list(range(100, 130))
duplicates = data.loc[duplicate_source_indices].copy()
data = pd.concat([data, duplicates], ignore_index=True)

ground_truth["injected_issues"]["duplicate_rows"] = {
    "source_rows": duplicate_source_indices,
    "expected_count": len(duplicate_source_indices)
}

ground_truth["final_rows"] = len(data)
ground_truth["final_columns"] = len(data.columns)

data.to_csv(OUTPUT_DIR / "synthetic_data.csv", index=False)
data.to_excel(OUTPUT_DIR / "synthetic_data.xlsx", index=False)

with open(OUTPUT_DIR / "ground_truth.json", "w", encoding="utf-8") as file:
    json.dump(ground_truth, file, indent=4)

print("Synthetic evaluation dataset created.")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")
print(f"CSV: {OUTPUT_DIR / 'synthetic_data.csv'}")
print(f"Excel: {OUTPUT_DIR / 'synthetic_data.xlsx'}")
print(f"Ground truth: {OUTPUT_DIR / 'ground_truth.json'}")