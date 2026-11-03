
import pandas as pd

from datadoctor.profiler import DataProfiler
from datadoctor.detectors import DataQualityDetector


def test_missing_value_detection():
    df = pd.DataFrame({
        "Age": [20, 25, None, 30, None]
    })

    profile = DataProfiler().profile_dataframe(df)
    report = DataQualityDetector().detect(df, profile)

    issue = next(
        issue for issue in report.issues
        if issue.issue_type == "missing_values"
    )

    assert issue.affected_records == 2


def test_duplicate_detection():
    df = pd.DataFrame({
        "Name": ["A", "B", "A"],
        "Age": [20, 25, 20]
    })

    profile = DataProfiler().profile_dataframe(df)
    report = DataQualityDetector().detect(df, profile)

    issue = next(
        issue for issue in report.issues
        if issue.issue_type == "duplicate_rows"
    )

    assert issue.affected_records == 1


def test_outlier_detection():
    df = pd.DataFrame({
        "Age": [20, 21, 22, 23, 24, 100]
    })

    profile = DataProfiler().profile_dataframe(df)
    report = DataQualityDetector().detect(df, profile)

    assert any(
        issue.issue_type == "potential_outliers"
        for issue in report.issues
    )


def test_categorical_formatting_detection():
    df = pd.DataFrame({
        "Gender": ["Male", "male", " FEMALE ", "Female"]
    })

    profile = DataProfiler().profile_dataframe(df)
    report = DataQualityDetector().detect(df, profile)

    assert any(
        issue.issue_type == "inconsistent_categorical_formatting"
        for issue in report.issues
    )


def test_detector_does_not_modify_dataframe():
    df = pd.DataFrame({
        "Age": [20, 21, 22, 100],
        "Gender": ["Male", "male", "Female", "Female"]
    })

    original = df.copy(deep=True)

    profile = DataProfiler().profile_dataframe(df)
    DataQualityDetector().detect(df, profile)

    pd.testing.assert_frame_equal(df, original)
