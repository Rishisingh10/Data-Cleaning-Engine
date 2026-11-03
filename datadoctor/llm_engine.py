
import json
import os
import time
from pathlib import Path

from pydantic import ValidationError

from datadoctor.schemas import (
    CleaningPlan,
    DatasetProfile,
    DataQualityReport
)


class GeminiEngineError(Exception):
    pass


def load_environment():
    env_path = Path(__file__).resolve().parent.parent / ".env"

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        if not env_path.exists():
            return

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            if key:
                os.environ.setdefault(key, value)


load_environment()


class GeminiCleaningEngine:

    RETRY_DELAYS = [0, 3, 7]

    NUMERIC_PHYSICAL_TYPES = {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Float32",
        "Float64"
    }

    IDENTIFIER_TYPES = {
        "identifier",
        "identifier_candidate"
    }

    OUTLIER_ISSUE_TYPES = {
        "outliers"
    }

    FORMATTING_ISSUE_TYPES = {
        "formatting_inconsistency"
    }

    ALLOWED_OPERATIONS = {
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

    def __init__(
        self,
        api_key=None,
        model=None,
        client=None,
        types_module=None
    ):
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY", "")
        ).strip()

        self.model = (
            model
            or os.getenv("GEMINI_MODEL")
            or "gemini-2.5-flash"
        ).strip()

        if client is not None:
            self.client = client
            self.types = types_module

            if self.types is None:
                try:
                    from google.genai import types
                    self.types = types
                except ImportError as error:
                    raise GeminiEngineError(
                        "Google Gemini SDK types are unavailable. "
                        "Install google-genai."
                    ) from error

            return

        if not self.api_key:
            raise GeminiEngineError(
                "GEMINI_API_KEY is not configured. "
                "Add GEMINI_API_KEY=your_key to the project .env file."
            )

        try:
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=60000)
            )

            self.types = types

        except ImportError as error:
            raise GeminiEngineError(
                "Google Gemini SDK is not installed. "
                "Run: pip install google-genai"
            ) from error

        except Exception as error:
            raise GeminiEngineError(
                f"Could not initialize Gemini client: {error}"
            ) from error

    @staticmethod
    def _build_context(
        profile: DatasetProfile,
        report: DataQualityReport,
        redacted_columns=None
    ):
        redacted_columns = {
            str(column)
            for column in (redacted_columns or [])
        }

        columns = []

        for column in profile.column_profiles.values():
            is_redacted = (
                column.column_name in redacted_columns
            )

            columns.append({
                "name": column.column_name,
                "physical_type": column.physical_type,
                "semantic_type": column.semantic_type,
                "semantic_confidence": column.semantic_confidence,
                "statistics": (
                    {"redacted": True}
                    if is_redacted
                    else column.statistics
                ),
                "missing_count": column.missing_count,
                "unique_count": column.unique_count,
                "inference_evidence": (
                    {"redacted": True}
                    if is_redacted
                    else column.inference_evidence
                )
            })

        issues = []

        for issue in report.issues:
            issue_data = issue.model_dump(mode="json")

            if issue.column_name in redacted_columns:
                issue_data["evidence"] = {
                    "redacted": True
                }

            issues.append(issue_data)

        return {
            "dataset": {
                "file_name": profile.file_name,
                "sheet_name": profile.sheet_name,
                "total_records": profile.total_records,
                "total_columns": profile.total_columns
            },
            "columns": columns,
            "issues": issues
        }

    @staticmethod
    def _prepare_schema(schema):
        if isinstance(schema, dict):
            return {
                key: GeminiCleaningEngine._prepare_schema(value)
                for key, value in schema.items()
                if key != "additionalProperties"
            }

        if isinstance(schema, list):
            return [
                GeminiCleaningEngine._prepare_schema(item)
                for item in schema
            ]

        return schema

    @staticmethod
    def _is_numeric_column(column):
        return (
            column.semantic_type == "numerical"
            and column.physical_type
            in GeminiCleaningEngine.NUMERIC_PHYSICAL_TYPES
        )

    @staticmethod
    def _get_parameters(recommendation):
        return recommendation.parameters or {}

    @classmethod
    def _validate_operation(
        cls,
        recommendation,
        issue,
        profile
    ):
        operation = recommendation.operation
        parameters = cls._get_parameters(recommendation)
        column_name = issue.column_name

        column = None

        if column_name is not None:
            column = next(
                (
                    item
                    for item in profile.column_profiles.values()
                    if item.column_name == column_name
                ),
                None
            )

        if operation not in cls.ALLOWED_OPERATIONS:
            raise GeminiEngineError(
                f"Unsupported cleaning operation: {operation}"
            )

        if operation == "leave_unchanged":
            if parameters:
                raise GeminiEngineError(
                    "leave_unchanged must not contain parameters."
                )
            return

        if operation == "remove_exact_duplicates":
            if issue.issue_type != "duplicate_rows":
                raise GeminiEngineError(
                    "Duplicate removal requires a duplicate_rows issue."
                )

            if column_name is not None or parameters:
                raise GeminiEngineError(
                    "Duplicate removal must target the dataset "
                    "and must not contain parameters."
                )
            return

        missing_value_operations = {
            "fill_mean",
            "fill_median",
            "fill_mode",
            "fill_constant",
            "drop_missing_rows"
        }

        if operation in missing_value_operations:
            if issue.issue_type != "missing_values":
                raise GeminiEngineError(
                    "Missing-value operations require a "
                    "missing_values issue."
                )

            if column is None:
                raise GeminiEngineError(
                    "Missing-value operation requires a valid target column."
                )

            if column.semantic_type in cls.IDENTIFIER_TYPES:
                raise GeminiEngineError(
                    "Missing-value operations are not allowed on identifiers."
                )

            if operation in {"fill_mean", "fill_median"}:
                if not cls._is_numeric_column(column):
                    raise GeminiEngineError(
                        f"{operation} is only allowed for numerical columns."
                    )

                if parameters:
                    raise GeminiEngineError(
                        f"{operation} must not contain parameters."
                    )

            elif operation == "fill_mode":
                if parameters:
                    raise GeminiEngineError(
                        "fill_mode must not contain parameters."
                    )

            elif operation == "fill_constant":
                if set(parameters) != {"value"}:
                    raise GeminiEngineError(
                        "fill_constant requires exactly one parameter: value."
                    )

            elif operation == "drop_missing_rows":
                if parameters:
                    raise GeminiEngineError(
                        "drop_missing_rows must not contain parameters."
                    )

            return

        formatting_operations = {
            "normalize_whitespace",
            "standardize_case"
        }

        if operation in formatting_operations:
            if issue.issue_type not in cls.FORMATTING_ISSUE_TYPES:
                raise GeminiEngineError(
                    "Formatting operations require a supported "
                    "formatting_inconsistency issue."
                )

            if column is None:
                raise GeminiEngineError(
                    "Formatting operation requires a valid target column."
                )

            if column.semantic_type in cls.IDENTIFIER_TYPES:
                raise GeminiEngineError(
                    "Formatting operations are not allowed on identifiers."
                )

            if operation == "normalize_whitespace":
                if parameters:
                    raise GeminiEngineError(
                        "normalize_whitespace must not contain parameters."
                    )

            if operation == "standardize_case":
                if set(parameters) != {"case"}:
                    raise GeminiEngineError(
                        "standardize_case requires exactly one parameter: case."
                    )

                if parameters["case"] not in {
                    "lower",
                    "upper",
                    "title"
                }:
                    raise GeminiEngineError(
                        "standardize_case must use lower, upper or title."
                    )

            return

        if operation == "review_outliers":
            if issue.issue_type not in cls.OUTLIER_ISSUE_TYPES:
                raise GeminiEngineError(
                    "Outlier review requires a supported outlier issue."
                )

            if parameters:
                raise GeminiEngineError(
                    "review_outliers must not contain parameters."
                )
            return

        raise GeminiEngineError(
            f"Unsupported cleaning operation: {operation}"
        )

    @classmethod
    def _validate_plan(
        cls,
        plan: CleaningPlan,
        report: DataQualityReport,
        profile: DatasetProfile
    ):
        valid_issues = {
            issue.issue_id: issue
            for issue in report.issues
        }

        seen_issue_ids = set()

        for recommendation in plan.recommendations:
            issue_id = recommendation.issue_id

            if issue_id not in valid_issues:
                raise GeminiEngineError(
                    f"Unknown issue ID: {issue_id}"
                )

            if issue_id in seen_issue_ids:
                raise GeminiEngineError(
                    f"Duplicate recommendation for issue: {issue_id}"
                )

            seen_issue_ids.add(issue_id)

            if not recommendation.requires_approval:
                raise GeminiEngineError(
                    "Every recommendation must require user approval."
                )

            issue = valid_issues[issue_id]

            if recommendation.target_column != issue.column_name:
                raise GeminiEngineError(
                    "Recommendation target does not match "
                    "the detected issue."
                )

            cls._validate_operation(
                recommendation,
                issue,
                profile
            )

    @staticmethod
    def _is_retryable(error):
        error_message = str(error).upper()

        retryable_codes = {
            "503",
            "UNAVAILABLE",
            "429",
            "RESOURCE_EXHAUSTED",
            "500",
            "INTERNAL",
            "DEADLINE_EXCEEDED",
            "TIMEOUT",
            "CONNECTION RESET",
            "TEMPORARILY UNAVAILABLE"
        }

        return any(
            code in error_message
            for code in retryable_codes
        )

    def _generate_response(self, prompt, schema):
        response = None
        last_error = None

        for attempt, delay in enumerate(self.RETRY_DELAYS):
            if delay:
                time.sleep(delay)

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        max_output_tokens=4096
                    )
                )
                break

            except Exception as error:
                last_error = error

                if not self._is_retryable(error):
                    raise GeminiEngineError(
                        f"Gemini API request failed: {error}"
                    ) from error

                if attempt == len(self.RETRY_DELAYS) - 1:
                    raise GeminiEngineError(
                        "Gemini API is unavailable after "
                        f"{len(self.RETRY_DELAYS)} attempts. "
                        "Please try again later."
                    ) from error

        if response is None:
            raise GeminiEngineError(
                f"Gemini did not return a response: {last_error}"
            )

        response_text = getattr(response, "text", None)

        if not response_text or not response_text.strip():
            raise GeminiEngineError(
                "Gemini returned an empty response. "
                "Try again or check the selected model."
            )

        return response_text.strip()

    def generate_cleaning_plan(
        self,
        profile: DatasetProfile,
        report: DataQualityReport,
        redacted_columns=None
    ) -> CleaningPlan:

        if profile != report.dataset_profile:
            raise GeminiEngineError(
                "The supplied profile and quality report do not match."
            )

        context = self._build_context(
            profile,
            report,
            redacted_columns
        )

        prompt = f"""
You are the DataDoctor AI cleaning recommendation engine.

Analyze the supplied dataset profile and detected quality issues.

DATASET INFORMATION:
{json.dumps(context, ensure_ascii=False, default=str)}

IMPORTANT RULES:

1. Recommend actions only for issues present in the supplied report.
2. Never invent issue IDs, columns, statistics or dataset facts.
3. Never assume every missing value should be imputed.
4. Never delete valid observations without justification.
5. Consider the semantic meaning and physical type of each column.
6. Never recommend mean or median imputation for identifiers,
   categorical columns, currency columns or free-form text.
7. Statistical outliers are not automatically incorrect.
8. Never automatically remove, cap or replace outliers.
9. For potential_outliers issues, recommend leave_unchanged.
10. For invalid_dates issues, recommend leave_unchanged.
11. For mixed_data_types issues, recommend leave_unchanged
    unless a supported operation is clearly valid.
12. For inconsistent_categorical_formatting issues,
    recommend leave_unchanged because the current execution
    engine does not support formatting operations for this issue type.
13. Use review_outliers only when the issue type is exactly "outliers".
14. Use formatting operations only when the issue type is exactly
    "formatting_inconsistency".
15. Use only the allowed operations listed below.
16. The issue_id must exactly match the supplied issue.
17. The target_column must exactly match the issue's column_name.
18. Never generate Python code, SQL, shell commands or executable scripts.
19. Explain the reasoning and risks of every recommendation.
20. Every recommendation requires user approval.
21. Do not invent operation parameters.
22. Use an empty parameters object when no parameters are needed.
23. fill_constant requires exactly one parameter named "value".
24. standardize_case requires exactly one parameter named "case".
25. The case parameter must be "lower", "upper" or "title".
26. remove_exact_duplicates must target the dataset, not a column.
27. If the correct action is uncertain, recommend leave_unchanged.
28. Do not generate more than one recommendation for the same issue.
29. Never recommend an operation incompatible with the column's
    semantic type or physical type.
30. It is acceptable to recommend leave_unchanged for an issue.

ALLOWED OPERATIONS:

- leave_unchanged
- drop_missing_rows
- fill_mean
- fill_median
- fill_mode
- fill_constant
- normalize_whitespace
- standardize_case
- remove_exact_duplicates
- review_outliers

Return a structured cleaning plan.
"""

        schema = self._prepare_schema(
            CleaningPlan.model_json_schema()
        )

        response_text = self._generate_response(
            prompt,
            schema
        )

        try:
            plan = CleaningPlan.model_validate_json(
                response_text
            )

        except ValidationError as error:
            raise GeminiEngineError(
                f"Gemini returned an invalid cleaning plan: {error}"
            ) from error

        except Exception as error:
            raise GeminiEngineError(
                f"Could not parse Gemini response: {error}"
            ) from error

        self._validate_plan(
            plan,
            report,
            profile
        )

        return plan
