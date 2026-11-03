
import pandas as pd
import pytest

from datadoctor.cleaning_executor import (
    CleaningExecutor,
    CleaningExecutionError
)
from datadoctor.profiler import DataProfiler
from datadoctor.schemas import (
    CleaningPlan,
    CleaningRecommendation,
    DataQualityIssue,
    DataQualityReport
)


@pytest.fixture
def executor():
    return CleaningExecutor()


def build_context(
    dataframe,
    issue_type,
    column,
    operation,
    parameters=None,
    issue_id="TEST-001"
):
    profile = DataProfiler().profile_dataframe(
        dataframe,
        file_name="test.csv"
    )

    issue = DataQualityIssue(
        issue_id=issue_id,
        issue_type=issue_type,
        column_name=column,
        severity="low",
        confidence=1.0,
        affected_records=1,
        evidence={},
        suggested_action=operation
    )

    report = DataQualityReport(
        dataset_profile=profile,
        issues=[issue]
    )

    recommendation = CleaningRecommendation(
        issue_id=issue_id,
        operation=operation,
        target_column=column,
        parameters=parameters or {},
        explanation="Test recommendation",
        rationale="Test rationale",
        expected_benefit="Test benefit",
        requires_approval=True
    )

    plan = CleaningPlan(
        dataset_summary="Test dataset",
        overall_assessment="Test assessment",
        recommendations=[recommendation]
    )

    return profile, report, plan


def execute(
    executor,
    dataframe,
    issue_type,
    column,
    operation,
    parameters=None
):
    profile, report, plan = build_context(
        dataframe,
        issue_type,
        column,
        operation,
        parameters
    )

    return executor.execute(
        dataframe,
        profile,
        report,
        plan,
        {"TEST-001"}
    )


def test_fill_median(executor):
    dataframe = pd.DataFrame({"Rating": [2.0, 4.0, None]})

    result, log = execute(
        executor,
        dataframe,
        "missing_values",
        "Rating",
        "fill_median"
    )

    assert result["Rating"].tolist() == [2.0, 4.0, 3.0]
    assert result["Rating"].isna().sum() == 0
    assert log[0]["missing_before"] == 1
    assert log[0]["missing_after"] == 0


def test_fill_mean(executor):
    dataframe = pd.DataFrame({"Score": [2.0, 4.0, None]})

    result, _ = execute(
        executor,
        dataframe,
        "missing_values",
        "Score",
        "fill_mean"
    )

    assert result["Score"].iloc[2] == 3.0


def test_fill_mode(executor):
    dataframe = pd.DataFrame({"City": ["Delhi", "Delhi", None]})

    result, _ = execute(
        executor,
        dataframe,
        "missing_values",
        "City",
        "fill_mode"
    )

    assert result["City"].isna().sum() == 0
    assert result["City"].iloc[2] == "Delhi"


def test_fill_constant(executor):
    dataframe = pd.DataFrame({"City": ["Delhi", None]})

    result, _ = execute(
        executor,
        dataframe,
        "missing_values",
        "City",
        "fill_constant",
        {"value": "Unknown"}
    )

    assert result["City"].tolist() == ["Delhi", "Unknown"]


def test_drop_missing_rows(executor):
    dataframe = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Score": [10, None, 30]
    })

    result, log = execute(
        executor,
        dataframe,
        "missing_values",
        "Score",
        "drop_missing_rows"
    )

    assert len(result) == 2
    assert log[0]["rows_removed"] == 1


def test_remove_exact_duplicates(executor):
    dataframe = pd.DataFrame({
        "Name": ["A", "B", "A"],
        "Score": [10, 20, 10]
    })

    result, log = execute(
        executor,
        dataframe,
        "duplicate_rows",
        None,
        "remove_exact_duplicates"
    )

    assert len(result) == 2
    assert log[0]["rows_removed"] == 1


def test_normalize_whitespace(executor):
    dataframe = pd.DataFrame({
        "City": [" New   Delhi ", "Mumbai"]
    })

    result, _ = execute(
        executor,
        dataframe,
        "formatting_inconsistency",
        "City",
        "normalize_whitespace"
    )

    assert result["City"].tolist() == ["New Delhi", "Mumbai"]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("lower", "delhi"),
        ("upper", "DELHI"),
        ("title", "Delhi")
    ]
)
def test_standardize_case(executor, case, expected):
    dataframe = pd.DataFrame({"City": ["delhi"]})

    result, _ = execute(
        executor,
        dataframe,
        "formatting_inconsistency",
        "City",
        "standardize_case",
        {"case": case}
    )

    assert result["City"].iloc[0] == expected


def test_review_outliers_does_not_change_data(executor):
    dataframe = pd.DataFrame({"Amount": [10, 20, 1000]})

    result, log = execute(
        executor,
        dataframe,
        "outliers",
        "Amount",
        "review_outliers"
    )

    pd.testing.assert_frame_equal(result, dataframe)
    assert "retained" in log[0]["action"]


def test_leave_unchanged(executor):
    dataframe = pd.DataFrame({"Rating": [3.0, 4.0]})

    result, _ = execute(
        executor,
        dataframe,
        "missing_values",
        "Rating",
        "leave_unchanged"
    )

    pd.testing.assert_frame_equal(result, dataframe)


def test_original_dataframe_is_never_modified(executor):
    dataframe = pd.DataFrame({"Rating": [2.0, 4.0, None]})
    original = dataframe.copy(deep=True)

    execute(
        executor,
        dataframe,
        "missing_values",
        "Rating",
        "fill_median"
    )

    pd.testing.assert_frame_equal(dataframe, original)


def test_unapproved_recommendation_is_rejected(executor):
    dataframe = pd.DataFrame({"Rating": [2.0, None]})

    profile, report, plan = build_context(
        dataframe,
        "missing_values",
        "Rating",
        "fill_median"
    )

    with pytest.raises(CleaningExecutionError):
        executor.execute(
            dataframe,
            profile,
            report,
            plan,
            set()
        )


def test_unknown_issue_id_is_rejected(executor):
    dataframe = pd.DataFrame({"Rating": [2.0, None]})

    profile, report, plan = build_context(
        dataframe,
        "missing_values",
        "Rating",
        "fill_median"
    )

    with pytest.raises(CleaningExecutionError):
        executor.execute(
            dataframe,
            profile,
            report,
            plan,
            {"UNKNOWN-ID"}
        )


def test_invalid_median_on_text_column_is_rejected(executor):
    dataframe = pd.DataFrame({"City": ["Delhi", "Mumbai", None]})

    profile, report, plan = build_context(
        dataframe,
        "missing_values",
        "City",
        "fill_median"
    )

    with pytest.raises(CleaningExecutionError):
        executor.execute(
            dataframe,
            profile,
            report,
            plan,
            {"TEST-001"}
        )


def test_invalid_case_parameter_is_rejected(executor):
    dataframe = pd.DataFrame({"City": ["Delhi", "Mumbai"]})

    profile, report, plan = build_context(
        dataframe,
        "formatting_inconsistency",
        "City",
        "standardize_case",
        {"case": "random"}
    )

    with pytest.raises(CleaningExecutionError):
        executor.execute(
            dataframe,
            profile,
            report,
            plan,
            {"TEST-001"}
        )
