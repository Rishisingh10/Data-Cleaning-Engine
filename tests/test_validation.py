
import pandas as pd

from datadoctor.validation import CleaningValidator


def test_validation_passes_for_unchanged_data():
    dataframe = pd.DataFrame({
        "Name": ["A", "B"],
        "Age": [20, 30]
    })

    result = CleaningValidator().validate(
        dataframe,
        dataframe.copy()
    )

    assert result["overall_status"] == "PASSED"
    assert result["failed"] == 0


def test_validation_detects_changed_columns():
    original = pd.DataFrame({
        "Name": ["A", "B"],
        "Age": [20, 30]
    })

    cleaned = original[["Age", "Name"]].copy()

    result = CleaningValidator().validate(
        original,
        cleaned
    )

    assert result["overall_status"] == "FAILED"


def test_validation_detects_dtype_changes():
    original = pd.DataFrame({
        "Age": [20, 30]
    })

    cleaned = pd.DataFrame({
        "Age": ["20", "30"]
    })

    result = CleaningValidator().validate(
        original,
        cleaned
    )

    assert result["overall_status"] == "FAILED"


def test_validation_detects_missing_value_increase():
    original = pd.DataFrame({
        "Age": [20, 30, 40]
    })

    cleaned = pd.DataFrame({
        "Age": [20, None, 40]
    })

    result = CleaningValidator().validate(
        original,
        cleaned
    )

    missing_check = next(
        item for item in result["checks"]
        if item["check"] == "Missing values"
    )

    assert missing_check["status"] == "WARNING"


def test_validation_accepts_recorded_row_removal():
    original = pd.DataFrame({
        "Name": ["A", "B", "C"]
    })

    cleaned = pd.DataFrame({
        "Name": ["A", "C"]
    })

    audit_log = [
        {"operation": "drop_missing_rows"}
    ]

    result = CleaningValidator().validate(
        original,
        cleaned,
        audit_log=audit_log
    )

    row_check = next(
        item for item in result["checks"]
        if item["check"] == "Row count"
    )

    assert row_check["status"] == "PASSED"


def test_validation_warns_about_unrecorded_row_removal():
    original = pd.DataFrame({
        "Name": ["A", "B", "C"]
    })

    cleaned = pd.DataFrame({
        "Name": ["A", "C"]
    })

    result = CleaningValidator().validate(
        original,
        cleaned
    )

    row_check = next(
        item for item in result["checks"]
        if item["check"] == "Row count"
    )

    assert row_check["status"] == "WARNING"


def test_validation_does_not_modify_dataframes():
    original = pd.DataFrame({
        "Name": [" A ", "B"],
        "Age": [20, 30]
    })

    cleaned = original.copy(deep=True)

    original_before = original.copy(deep=True)
    cleaned_before = cleaned.copy(deep=True)

    CleaningValidator().validate(original, cleaned)

    pd.testing.assert_frame_equal(original, original_before)
    pd.testing.assert_frame_equal(cleaned, cleaned_before)
