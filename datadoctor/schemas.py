
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_name: str
    physical_type: str
    semantic_type: str = "unknown"
    semantic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    inference_evidence: list[str] = Field(default_factory=list)

    total_records: int = Field(ge=0)
    non_null_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)

    statistics: dict[str, Any] = Field(default_factory=dict)
    sample_values: list[Any] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.non_null_count + self.missing_count != self.total_records:
            raise ValueError("Inconsistent record counts.")

        if self.unique_count > self.non_null_count:
            raise ValueError("Unique count exceeds non-null count.")

        return self


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    sheet_name: str | None = None
    total_records: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    column_profiles: dict[str, ColumnProfile] = Field(default_factory=dict)
    profiled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_dataset(self):
        if self.total_columns != len(self.column_profiles):
            raise ValueError("Column count mismatch.")

        if any(
            profile.total_records != self.total_records
            for profile in self.column_profiles.values()
        ):
            raise ValueError("Record count mismatch.")

        return self


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    issue_type: str
    column_name: str | None = None
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    affected_records: int = Field(ge=0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    suggested_action: str
    requires_user_review: bool = True


class DataQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_profile: DatasetProfile
    issues: list[DataQualityIssue] = Field(default_factory=list)

    @computed_field
    @property
    def total_issues(self) -> int:
        return len(self.issues)

    @computed_field
    @property
    def severity_summary(self) -> dict[str, int]:
        summary = dict.fromkeys(
            ["low", "medium", "high", "critical"], 0
        )

        for issue in self.issues:
            summary[issue.severity] += 1

        return summary

class CleaningRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    operation: Literal[
        "leave_unchanged",
        "drop_missing_rows",
        "fill_mean",
        "fill_median",
        "fill_mode",
        "fill_constant",
        "normalize_whitespace",
        "standardize_case",
        "remove_exact_duplicates",
        "review_outliers"
    ]

    target_column: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    explanation: str
    rationale: str
    expected_benefit: str
    risks: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)

    requires_approval: bool = True


class CleaningPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_summary: str
    overall_assessment: str

    recommendations: list[CleaningRecommendation] = Field(
        default_factory=list
    )

    important_warnings: list[str] = Field(
        default_factory=list
    )
