import pandas as pd

from datadoctor.schema_drift import SchemaDriftDetector


def test_detects_added_and_removed_columns():
    previous = pd.DataFrame({
        "customer_id": [1],
        "name": ["A"]
    })

    current = pd.DataFrame({
        "customer_id": [1],
        "email": ["a@example.com"]
    })

    result = SchemaDriftDetector().compare(
        previous,
        current
    )

    assert result["added_columns"] == ["email"]
    assert result["removed_columns"] == ["name"]
    assert result["has_drift"]


def test_detects_dtype_and_semantic_drift():
    previous = pd.DataFrame({
        "amount": [10, 20, 30]
    })

    current = pd.DataFrame({
        "amount": ["low", "medium", "high"]
    })

    result = SchemaDriftDetector().compare(
        previous,
        current
    )

    assert result["dtype_changes"]
    assert result["semantic_type_changes"]


def test_same_schema_has_no_drift():
    frame = pd.DataFrame({
        "name": ["A", "B"],
        "age": [20, 30]
    })

    result = SchemaDriftDetector().compare(
        frame,
        frame.copy()
    )

    assert result["has_drift"] is False
    assert result["row_count_difference"] == 0