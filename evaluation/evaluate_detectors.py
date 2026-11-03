
import sys
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datadoctor.profiler import DataProfiler
from datadoctor.detectors import DataQualityDetector


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "test_datasets" / "synthetic_data.csv"
GROUND_TRUTH_PATH = BASE_DIR / "test_datasets" / "ground_truth.json"


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_PATH}")

    dataframe = pd.read_csv(DATASET_PATH)

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    return dataframe, ground_truth


def generate_report(dataframe):
    profiler = DataProfiler()

    profile = profiler.profile_dataframe(
        dataframe,
        file_name=DATASET_PATH.name,
        sheet_name="Sheet1"
    )

    detector = DataQualityDetector()
    report = detector.detect(dataframe, profile)

    return profile, report


def display_dataset_information(dataframe):
    print("\nDATASET INFORMATION")
    print("=" * 60)
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Column names: {list(dataframe.columns)}")


def display_detection_results(report):
    print("\nDETECTION RESULTS")
    print("=" * 60)

    detected_types = set()

    for issue in report.issues:
        detected_types.add(issue.issue_type)

        print(f"Issue type       : {issue.issue_type}")
        print(f"Column           : {issue.column_name}")
        print(f"Severity         : {issue.severity}")
        print(f"Affected records : {issue.affected_records}")
        print("-" * 40)

    return detected_types


def evaluate_issue_types(detected_types, ground_truth):
    print("\nEVALUATION SUMMARY")
    print("=" * 60)

    expected_types = set(ground_truth["injected_issues"].keys())

    passed = 0
    failed = 0

    for issue_type, details in ground_truth["injected_issues"].items():
        detected = issue_type in detected_types
        status = "PASS" if detected else "FAIL"

        print(
            f"{status} | {issue_type} | "
            f"Expected injected count: {details['expected_count']}"
        )

        if detected:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"Expected issue types : {len(expected_types)}")
    print(f"Detected issue types : {passed}")
    print(f"Missed issue types   : {failed}")

    if passed + failed:
        print(f"Detection coverage   : {passed / (passed + failed) * 100:.2f}%")

    return failed == 0


def main():
    dataframe, ground_truth = load_dataset()

    display_dataset_information(dataframe)

    profile, report = generate_report(dataframe)

    detected_types = display_detection_results(report)

    all_detected = evaluate_issue_types(detected_types, ground_truth)

    print("\nFINAL RESULT")
    print("=" * 60)

    if all_detected:
        print("All expected issue types were detected.")
    else:
        print("Some expected issue types were not detected.")

    print("\nNote: This evaluation checks issue-type coverage only.")
    print("It does not yet calculate row-level precision or recall.")


if __name__ == "__main__":
    main()
