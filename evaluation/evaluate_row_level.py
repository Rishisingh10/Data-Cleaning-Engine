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
RESULTS_PATH = BASE_DIR / "row_level_results.csv"


def load_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(GROUND_TRUTH_PATH)

    dataframe = pd.read_csv(DATASET_PATH)

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    return dataframe, ground_truth


def run_detector(dataframe):
    profiler = DataProfiler()

    profile = profiler.profile_dataframe(
        dataframe,
        file_name=DATASET_PATH.name,
        sheet_name="Sheet1"
    )

    detector = DataQualityDetector()

    return detector.detect(dataframe, profile)


def get_expected_positions(issue_type, details, total_rows):
    if issue_type == "missing_values":
        return set(details["affected_rows"])

    if issue_type == "inconsistent_categorical_formatting":
        return set(details["affected_rows"])

    if issue_type == "potential_outliers":
        return set(details["affected_rows"])

    if issue_type == "duplicate_rows":
        if "duplicate_row_positions" in details:
            return set(details["duplicate_row_positions"])

        duplicate_count = details["expected_count"]

        return set(
            range(total_rows - duplicate_count, total_rows)
        )

    raise ValueError(f"Unsupported issue type: {issue_type}")


def get_predicted_positions(report, issue_type):
    predicted_positions = set()

    for issue in report.issues:
        if issue.issue_type != issue_type:
            continue

        positions = issue.evidence.get(
            "affected_row_positions",
            []
        )

        predicted_positions.update(
            int(position) for position in positions
        )

    return predicted_positions


def calculate_metrics(expected, predicted, total_rows):
    all_positions = set(range(total_rows))

    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)

    true_negatives = len(
        all_positions - (expected | predicted)
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    accuracy = (
        (true_positives + true_negatives) / total_rows
        if total_rows else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1_score * 100, 2),
        "accuracy": round(accuracy * 100, 2)
    }


def main():
    dataframe, ground_truth = load_data()

    print("\nDATASET INFORMATION")
    print("=" * 65)
    print(f"Dataset: {DATASET_PATH.name}")
    print(f"Total rows: {len(dataframe)}")
    print(f"Total columns: {len(dataframe.columns)}")

    report = run_detector(dataframe)

    results = []

    print("\nROW-LEVEL DETECTION EVALUATION")
    print("=" * 65)

    for issue_type, details in ground_truth["injected_issues"].items():

        expected = get_expected_positions(
            issue_type,
            details,
            len(dataframe)
        )

        predicted = get_predicted_positions(
            report,
            issue_type
        )

        invalid_positions = predicted - set(range(len(dataframe)))

        if invalid_positions:
            raise ValueError(
                f"{issue_type} returned invalid row positions: "
                f"{sorted(invalid_positions)}"
            )

        metrics = calculate_metrics(
            expected,
            predicted,
            len(dataframe)
        )

        result = {
            "issue_type": issue_type,
            "expected_records": len(expected),
            "detected_records": len(predicted),
            **metrics
        }

        results.append(result)

        print(f"\nDetector: {issue_type}")
        print("-" * 65)
        print(f"Expected records : {len(expected)}")
        print(f"Detected records : {len(predicted)}")
        print(f"True positives   : {metrics['true_positives']}")
        print(f"False positives  : {metrics['false_positives']}")
        print(f"False negatives  : {metrics['false_negatives']}")
        print(f"True negatives   : {metrics['true_negatives']}")
        print(f"Precision        : {metrics['precision']}%")
        print(f"Recall           : {metrics['recall']}%")
        print(f"F1-score         : {metrics['f1_score']}%")
        print(f"Accuracy         : {metrics['accuracy']}%")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\nOVERALL EVALUATION")
    print("=" * 65)

    print(
        "Macro Precision:",
        round(results_df["precision"].mean(), 2),
        "%"
    )

    print(
        "Macro Recall:",
        round(results_df["recall"].mean(), 2),
        "%"
    )

    print(
        "Macro F1-score:",
        round(results_df["f1_score"].mean(), 2),
        "%"
    )

    print("\nDetailed results saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()