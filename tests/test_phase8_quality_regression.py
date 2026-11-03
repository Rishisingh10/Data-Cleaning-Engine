import pandas as pd

from datadoctor.profiler import DataProfiler
from datadoctor.detectors import DataQualityDetector


def analyze(frame):
    profile = DataProfiler().profile_dataframe(frame)
    report = DataQualityDetector().detect(frame, profile)
    return profile, report


def test_missing_values_are_detected():
    frame = pd.DataFrame({
        "name": ["A", None, "C"]
    })

    _, report = analyze(frame)

    issue = next(
        item for item in report.issues
        if item.issue_type == "missing_values"
    )

    assert issue.affected_records == 1


def test_duplicate_records_are_detected():
    frame = pd.DataFrame({
        "name": ["A", "A", "B"],
        "age": [20, 20, 30]
    })

    _, report = analyze(frame)

    issue = next(
        item for item in report.issues
        if item.issue_type == "duplicate_rows"
    )

    assert issue.affected_records == 1


def test_identifier_does_not_receive_numeric_statistics():
    frame = pd.DataFrame({
        "customer_id": [1001, 1002, 1003, 1004]
    })

    profile, _ = analyze(frame)
    column = profile.column_profiles["customer_id"]

    assert column.semantic_type == "identifier"
    assert "mean" not in column.statistics
    assert "median" not in column.statistics
    assert "std" not in column.statistics


def test_mixed_data_types_are_detected():
    frame = pd.DataFrame({
        "measurement": [10, 20, 30, 40, "unknown"]
    })

    profile, report = analyze(frame)

    assert (
        profile.column_profiles["measurement"].semantic_type
        == "mixed"
    )

    issue = next(
        item for item in report.issues
        if item.issue_type == "mixed_data_types"
    )

    assert issue.affected_records == 1


def test_invalid_dates_are_detected():
    frame = pd.DataFrame({
        "order_date": [
            "2026-01-10",
            "not-a-date",
            "2026-02-01"
        ]
    })

    _, report = analyze(frame)

    issue = next(
        item for item in report.issues
        if item.issue_type == "invalid_dates"
    )

    assert issue.affected_records == 1

    assert (
        issue.evidence["sample_invalid_values"]
        == ["not-a-date"]
    )