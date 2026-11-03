
import pandas as pd

from datadoctor.schemas import (
    CleaningPlan,
    DatasetProfile,
    DataQualityReport
)
from datadoctor.integrity import DataIntegrity, DataIntegrityError


class CleaningExecutionError(Exception):
    pass


class CleaningExecutor:

    NUMERIC_TYPES = {"numerical"}
    IDENTIFIER_TYPES = {"identifier", "identifier_candidate"}

    SUPPORTED_OPERATIONS = {
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
    }

    def execute(
        self,
        dataframe: pd.DataFrame,
        profile: DatasetProfile,
        report: DataQualityReport,
        plan: CleaningPlan,
        approved_issue_ids: set[str],
        expected_fingerprint: str | None = None
    ) -> tuple[pd.DataFrame, list[dict]]:

        if not approved_issue_ids:
            raise CleaningExecutionError(
                "No recommendations have been approved."
            )

        if profile != report.dataset_profile:
            raise CleaningExecutionError(
                "Profile and quality report do not match."
            )

        if dataframe.shape != (
            profile.total_records,
            profile.total_columns
        ):
            raise CleaningExecutionError(
                "Dataset dimensions do not match the supplied profile."
            )

        if dataframe.columns.duplicated().any():
            raise CleaningExecutionError(
                "Duplicate column names are not supported."
            )

        if list(dataframe.columns) != [
            item.column_name
            for item in profile.column_profiles.values()
        ]:
            raise CleaningExecutionError(
                "Dataset columns do not match the supplied profile."
            )

        original_fingerprint = DataIntegrity.fingerprint(dataframe)

        if (
            expected_fingerprint is not None
            and original_fingerprint != expected_fingerprint
        ):
            raise CleaningExecutionError(
                "Dataset changed after analysis. Please analyze it again."
            )

        issues = {
            issue.issue_id: issue
            for issue in report.issues
        }

        recommendations = {
            item.issue_id: item
            for item in plan.recommendations
        }

        if len(recommendations) != len(plan.recommendations):
            raise CleaningExecutionError(
                "Duplicate issue recommendations found."
            )

        unknown_ids = approved_issue_ids - recommendations.keys()

        if unknown_ids:
            raise CleaningExecutionError(
                f"Unknown approved issue IDs: {sorted(unknown_ids)}"
            )

        result = dataframe.copy(deep=True)
        audit_log = []

        try:
            for issue_id in sorted(approved_issue_ids):
                recommendation = recommendations[issue_id]
                issue = issues.get(issue_id)

                if issue is None:
                    raise CleaningExecutionError(
                        f"No detected issue exists for {issue_id}."
                    )

                self._validate_recommendation(
                    result,
                    profile,
                    issue,
                    recommendation
                )

                before_rows = len(result)
                column = recommendation.target_column

                before_missing = (
                    int(result[column].isna().sum())
                    if column is not None
                    else None
                )

                result, action = self._apply(
                    result,
                    recommendation
                )

                after_rows = len(result)

                after_missing = (
                    int(result[column].isna().sum())
                    if column is not None
                    else None
                )

                audit_log.append({
                    "issue_id": issue_id,
                    "operation": recommendation.operation,
                    "target_column": column,
                    "rows_before": before_rows,
                    "rows_after": after_rows,
                    "rows_removed": before_rows - after_rows,
                    "missing_before": before_missing,
                    "missing_after": after_missing,
                    "action": action
                })

            DataIntegrity.validate_structure(
                dataframe,
                result
            )

            DataIntegrity.validate_original(
                dataframe,
                original_fingerprint
            )

            return result, audit_log

        except CleaningExecutionError:
            raise

        except Exception as error:
            raise CleaningExecutionError(
                f"Cleaning execution failed: {error}"
            ) from error

    def _validate_recommendation(
        self,
        dataframe,
        profile,
        issue,
        recommendation
    ):

        operation = recommendation.operation
        column = recommendation.target_column
        parameters = recommendation.parameters

        if operation not in self.SUPPORTED_OPERATIONS:
            raise CleaningExecutionError(
                f"Unsupported operation: {operation}"
            )

        if not recommendation.requires_approval:
            raise CleaningExecutionError(
                "Recommendation does not require approval."
            )

        if column != issue.column_name:
            raise CleaningExecutionError(
                "Recommendation target does not match the detected issue."
            )

        if column is not None and column not in dataframe.columns:
            raise CleaningExecutionError(
                f"Column not found: {column}"
            )

        if operation == "remove_exact_duplicates":
            if issue.issue_type != "duplicate_rows" or column is not None:
                raise CleaningExecutionError(
                    "Invalid duplicate-removal recommendation."
                )

            if parameters:
                raise CleaningExecutionError(
                    "Duplicate removal does not accept parameters."
                )

            return

        if operation == "leave_unchanged":
            if parameters:
                raise CleaningExecutionError(
                    "leave_unchanged does not accept parameters."
                )

            return

        if operation == "review_outliers":
            if issue.issue_type != "outliers":
                raise CleaningExecutionError(
                    "Outlier review requires an outlier issue."
                )

            if parameters:
                raise CleaningExecutionError(
                    "Outlier review does not accept parameters."
                )

            return

        if column is None:
            raise CleaningExecutionError(
                "This operation requires a target column."
            )

        column_profile = next(
            (
                item
                for item in profile.column_profiles.values()
                if item.column_name == column
            ),
            None
        )

        if column_profile is None:
            raise CleaningExecutionError(
                f"Column profile not found: {column}"
            )

        if operation in {
            "drop_missing_rows",
            "fill_mean",
            "fill_median",
            "fill_mode",
            "fill_constant"
        }:
            if issue.issue_type != "missing_values":
                raise CleaningExecutionError(
                    "Missing-value operation requires a missing-values issue."
                )

        if operation in {"fill_mean", "fill_median"}:
            if (
                column_profile.semantic_type not in self.NUMERIC_TYPES
                or not pd.api.types.is_numeric_dtype(dataframe[column])
            ):
                raise CleaningExecutionError(
                    "Mean and median imputation require a numerical column."
                )

            if parameters:
                raise CleaningExecutionError(
                    "Mean and median values are calculated by Python."
                )

        if operation == "fill_mode":
            if column_profile.semantic_type in self.IDENTIFIER_TYPES:
                raise CleaningExecutionError(
                    "Mode imputation is not allowed for identifier columns."
                )

            if parameters:
                raise CleaningExecutionError(
                    "Mode imputation does not accept parameters."
                )

        if operation == "fill_constant":
            if column_profile.semantic_type in self.IDENTIFIER_TYPES:
                raise CleaningExecutionError(
                    "Constant imputation is not allowed for identifier columns."
                )

            if set(parameters) != {"value"}:
                raise CleaningExecutionError(
                    "fill_constant requires only a value parameter."
                )

        if operation == "drop_missing_rows" and parameters:
            raise CleaningExecutionError(
                "drop_missing_rows does not accept parameters."
            )

        if operation in {
            "normalize_whitespace",
            "standardize_case"
        }:
            if issue.issue_type != "formatting_inconsistency":
                raise CleaningExecutionError(
                    "Formatting operation requires a formatting issue."
                )

            if not (
                pd.api.types.is_object_dtype(dataframe[column])
                or pd.api.types.is_string_dtype(dataframe[column])
            ):
                raise CleaningExecutionError(
                    "Formatting operations require a text column."
                )

        if operation == "normalize_whitespace" and parameters:
            raise CleaningExecutionError(
                "normalize_whitespace does not accept parameters."
            )

        if operation == "standardize_case":
            if set(parameters) != {"case"}:
                raise CleaningExecutionError(
                    "standardize_case requires only a case parameter."
                )

            if parameters["case"] not in {"lower", "upper", "title"}:
                raise CleaningExecutionError(
                    "Case must be lower, upper or title."
                )

    def _apply(self, dataframe, recommendation):

        operation = recommendation.operation
        column = recommendation.target_column
        parameters = recommendation.parameters

        if operation == "leave_unchanged":
            return dataframe, "No changes made."

        if operation == "review_outliers":
            return dataframe, "Outliers retained for manual review."

        if operation == "drop_missing_rows":
            before = len(dataframe)
            result = dataframe.dropna(subset=[column]).copy()
            removed = before - len(result)

            return result, f"Removed {removed} rows missing {column}."

        if operation == "fill_mean":
            value = dataframe[column].mean()

            if pd.isna(value):
                raise CleaningExecutionError(
                    f"Cannot calculate mean for {column}."
                )

            result = dataframe.copy()
            result[column] = result[column].fillna(value)

            return result, f"Missing values filled with mean: {value}."

        if operation == "fill_median":
            value = dataframe[column].median()

            if pd.isna(value):
                raise CleaningExecutionError(
                    f"Cannot calculate median for {column}."
                )

            result = dataframe.copy()
            result[column] = result[column].fillna(value)

            return result, f"Missing values filled with median: {value}."

        if operation == "fill_mode":
            modes = dataframe[column].mode(dropna=True)

            if modes.empty:
                raise CleaningExecutionError(
                    f"Cannot calculate mode for {column}."
                )

            value = modes.iloc[0]
            result = dataframe.copy()
            result[column] = result[column].fillna(value)

            return result, f"Missing values filled with mode: {value}."

        if operation == "fill_constant":
            value = parameters["value"]
            result = dataframe.copy()
            result[column] = result[column].fillna(value)

            return result, f"Missing values filled with constant: {value}."

        if operation == "normalize_whitespace":
            result = dataframe.copy()

            result[column] = result[column].map(
                lambda value: (
                    " ".join(value.split())
                    if isinstance(value, str)
                    else value
                )
            )

            return result, "Whitespace normalized."

        if operation == "standardize_case":
            case = parameters["case"]
            result = dataframe.copy()

            converters = {
                "lower": str.lower,
                "upper": str.upper,
                "title": str.title
            }

            converter = converters[case]

            result[column] = result[column].map(
                lambda value: (
                    converter(value)
                    if isinstance(value, str)
                    else value
                )
            )

            return result, f"Text converted to {case} case."

        if operation == "remove_exact_duplicates":
            before = len(dataframe)
            result = dataframe.drop_duplicates().copy()
            removed = before - len(result)

            return result, f"Removed {removed} exact duplicate rows."

        raise CleaningExecutionError(
            f"Unsupported operation: {operation}"
        )
