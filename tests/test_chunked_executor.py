import pandas as pd
import pytest

from datadoctor.chunked_executor import (
    ChunkedCleaningExecutor,
    ChunkedExecutionError
)

from datadoctor.profiler import DataProfiler
from datadoctor.detectors import DataQualityDetector

from datadoctor.schemas import (
    CleaningPlan,
    CleaningRecommendation
)


def make_plan_and_report(
    frame,
    issue_type,
    operation,
    column=None,
    parameters=None
):
    profile = DataProfiler().profile_dataframe(frame)

    report = DataQualityDetector().detect(
        frame,
        profile
    )

    issue = next(
        item for item in report.issues
        if item.issue_type == issue_type
    )

    recommendation = CleaningRecommendation(
        issue_id=issue.issue_id,
        operation=operation,
        target_column=column,
        parameters=parameters or {},
        explanation="Approved test operation",
        rationale="Testing chunk processing",
        expected_benefit="Validated chunk output",
        requires_approval=True
    )

    plan = CleaningPlan(
        dataset_summary="Test dataset",
        overall_assessment="Test plan",
        recommendations=[recommendation]
    )

    return plan, report, {issue.issue_id}


def test_removes_duplicates_across_chunk_boundaries(tmp_path):
    frame = pd.DataFrame({
        "name": ["A", "B", "A", "C"]
    })

    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"

    frame.to_csv(source, index=False)

    plan, report, approved = make_plan_and_report(
        frame,
        "duplicate_rows",
        "remove_exact_duplicates"
    )

    audit = ChunkedCleaningExecutor().execute_csv(
        source,
        output,
        plan,
        approved,
        report,
        chunksize=2
    )

    result = pd.read_csv(output)

    assert result["name"].tolist() == ["A", "B", "C"]
    assert audit["duplicates_removed"] == 1
    assert audit["rows_read"] == 4
    assert audit["rows_written"] == 3


def test_chunked_mean_uses_global_mean(tmp_path):
    frame = pd.DataFrame({
        "amount": [10.0, None, 30.0, None]
    })

    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"

    frame.to_csv(source, index=False)

    plan, report, approved = make_plan_and_report(
        frame,
        "missing_values",
        "fill_mean",
        "amount"
    )

    ChunkedCleaningExecutor().execute_csv(
        source,
        output,
        plan,
        approved,
        report,
        chunksize=2
    )

    result = pd.read_csv(output)

    assert result["amount"].tolist() == [
        10.0, 20.0, 30.0, 20.0
    ]


def test_rejects_unsupported_operations(tmp_path):
    frame = pd.DataFrame({
        "amount": [10.0, None, 30.0]
    })

    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"

    frame.to_csv(source, index=False)

    plan, report, approved = make_plan_and_report(
        frame,
        "missing_values",
        "fill_median",
        "amount"
    )

    with pytest.raises(
        ChunkedExecutionError,
        match="not supported"
    ):
        ChunkedCleaningExecutor().execute_csv(
            source,
            output,
            plan,
            approved,
            report,
            chunksize=2
        )

    assert not output.exists()


def test_large_csv_is_processed_in_multiple_chunks(tmp_path):
    source = tmp_path / "large.csv"
    output = tmp_path / "large_cleaned.csv"

    frame = pd.DataFrame({
        "name": [None] + [
            f"user_{i}" for i in range(1, 25000)
        ]
    })

    frame.to_csv(source, index=False)

    plan, report, approved = make_plan_and_report(
        frame,
        "missing_values",
        "drop_missing_rows",
        "name"
    )

    audit = ChunkedCleaningExecutor().execute_csv(
        source,
        output,
        plan,
        approved,
        report,
        chunksize=2000
    )

    assert audit["rows_read"] == 25000
    assert audit["rows_written"] == 24999
    assert len(pd.read_csv(output)) == 24999