
import re
import uuid

import pandas as pd

from datadoctor.schemas import (
    DataQualityIssue,
    DataQualityReport,
    DatasetProfile
)


class DataQualityDetector:

    NUMERIC_TYPES = {
        "numerical",
        "numerical_text",
        "currency"
    }

    DATE_NAME_PATTERN = re.compile(
        r"(^|_)(date|datetime|timestamp|time|dob|birth_date|"
        r"created_at|updated_at|joined_at|start_date|end_date)($|_)",
        re.IGNORECASE
    )

    def detect(
        self,
        dataframe: pd.DataFrame,
        profile: DatasetProfile
    ) -> DataQualityReport:

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        if len(dataframe) != profile.total_records:
            raise ValueError(
                "Dataset and profile record counts do not match."
            )

        if len(dataframe.columns) != profile.total_columns:
            raise ValueError(
                "Dataset and profile column counts do not match."
            )

        if dataframe.columns.duplicated().any():
            raise ValueError(
                "Duplicate column names are not supported."
            )

        issues = []

        for position, column in enumerate(dataframe.columns):
            series = dataframe.iloc[:, position]
            column_name = str(column)

            profile_key = column_name

            if profile_key not in profile.column_profiles:
                profile_key = f"{column_name}_{position + 1}"

            if profile_key not in profile.column_profiles:
                raise ValueError(
                    f"Column profile not found for: {column_name}"
                )

            column_profile = profile.column_profiles[profile_key]

            issues.extend(
                self._detect_missing_values(
                    series,
                    column_name
                )
            )

            issues.extend(
                self._detect_invalid_dates(
                    series,
                    column_name,
                    column_profile.semantic_type
                )
            )

            issues.extend(
                self._detect_outliers(
                    series,
                    column_name,
                    column_profile.semantic_type
                )
            )

            issues.extend(
                self._detect_mixed_types(
                    series,
                    column_name,
                    column_profile.semantic_type
                )
            )

            issues.extend(
                self._detect_formatting(
                    series,
                    column_name,
                    column_profile.semantic_type
                )
            )

        issues.extend(
            self._detect_duplicate_rows(dataframe)
        )

        return DataQualityReport(
            dataset_profile=profile,
            issues=issues
        )

    def _create_issue(
        self,
        issue_type: str,
        severity: str,
        confidence: float,
        affected_records: int,
        suggested_action: str,
        column_name: str | None = None,
        evidence: dict | None = None,
        requires_user_review: bool = True
    ) -> DataQualityIssue:

        return DataQualityIssue(
            issue_id=f"DQ-{uuid.uuid4().hex[:8].upper()}",
            issue_type=issue_type,
            column_name=column_name,
            severity=severity,
            confidence=confidence,
            affected_records=affected_records,
            evidence=evidence or {},
            suggested_action=suggested_action,
            requires_user_review=requires_user_review
        )

    @staticmethod
    def _severity(percentage: float) -> str:

        if percentage >= 20:
            return "critical"

        if percentage >= 10:
            return "high"

        if percentage >= 5:
            return "medium"

        return "low"

    @staticmethod
    def _row_positions(mask: pd.Series) -> list[int]:

        return [
            int(position)
            for position, matched in enumerate(mask.to_numpy())
            if matched
        ]

    @staticmethod
    def _missing_mask(series: pd.Series) -> pd.Series:

        mask = series.isna()

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            mask |= (
                series.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )

        return mask

    def _detect_missing_values(
        self,
        series: pd.Series,
        column_name: str
    ) -> list[DataQualityIssue]:

        missing_mask = self._missing_mask(series)
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            return []

        percentage = (
            missing_count / max(len(series), 1) * 100
        )

        return [
            self._create_issue(
                issue_type="missing_values",
                column_name=column_name,
                severity=self._severity(percentage),
                confidence=1.0,
                affected_records=missing_count,
                evidence={
                    "missing_count": missing_count,
                    "missing_percentage": round(percentage, 2),
                    "total_records": len(series),
                    "affected_row_positions": self._row_positions(
                        missing_mask
                    )
                },
                suggested_action=(
                    "Review missing values and choose an appropriate "
                    "strategy: imputation, removal or an explicit "
                    "unknown category."
                )
            )
        ]

    def _detect_duplicate_rows(
        self,
        dataframe: pd.DataFrame
    ) -> list[DataQualityIssue]:

        duplicate_mask = dataframe.duplicated(keep="first")
        duplicate_count = int(duplicate_mask.sum())

        if duplicate_count == 0:
            return []

        percentage = (
            duplicate_count / max(len(dataframe), 1) * 100
        )

        duplicate_positions = self._row_positions(duplicate_mask)

        sample_indices = (
            dataframe.index[duplicate_mask]
            .tolist()[:10]
        )

        return [
            self._create_issue(
                issue_type="duplicate_rows",
                severity=self._severity(percentage),
                confidence=1.0,
                affected_records=duplicate_count,
                evidence={
                    "duplicate_count": duplicate_count,
                    "duplicate_percentage": round(percentage, 2),
                    "sample_row_indices": [
                        str(index)
                        for index in sample_indices
                    ],
                    "affected_row_positions": duplicate_positions,
                    "comparison": "Exact full-row duplicates"
                },
                suggested_action=(
                    "Review duplicate records and confirm the dataset's "
                    "uniqueness rules before removing any rows."
                )
            )
        ]

    def _detect_invalid_dates(
        self,
        series: pd.Series,
        column_name: str,
        semantic_type: str
    ) -> list[DataQualityIssue]:

        is_date_column = (
            semantic_type == "datetime"
            or bool(self.DATE_NAME_PATTERN.search(column_name))
        )

        if not is_date_column:
            return []

        missing_mask = self._missing_mask(series)
        candidates = series.loc[~missing_mask]

        if candidates.empty:
            return []

        try:
            parsed = pd.to_datetime(
                candidates,
                format="mixed",
                errors="coerce",
                utc=True
            )

        except (TypeError, ValueError):
            parsed = pd.to_datetime(
                candidates,
                errors="coerce",
                utc=True
            )

        invalid_mask = parsed.isna()
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            return []

        percentage = (
            invalid_count / max(len(candidates), 1) * 100
        )

        invalid_values = (
            candidates.loc[invalid_mask]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        affected_positions = self._row_positions(
            series.index.to_series().isin(
                candidates.index[invalid_mask]
            )
        )

        return [
            self._create_issue(
                issue_type="invalid_dates",
                column_name=column_name,
                severity=self._severity(percentage),
                confidence=0.95,
                affected_records=invalid_count,
                evidence={
                    "invalid_date_count": invalid_count,
                    "invalid_date_percentage": round(
                        percentage,
                        2
                    ),
                    "valid_date_count": int(
                        len(candidates) - invalid_count
                    ),
                    "sample_invalid_values": invalid_values,
                    "affected_row_positions": affected_positions
                },
                suggested_action=(
                    "Review invalid date values and determine the "
                    "correct date format or replacement. Do not "
                    "automatically replace ambiguous dates."
                )
            )
        ]

    def _detect_outliers(
        self,
        series: pd.Series,
        column_name: str,
        semantic_type: str
    ) -> list[DataQualityIssue]:

        if semantic_type not in self.NUMERIC_TYPES:
            return []

        numeric = self._to_numeric(series)

        valid_mask = numeric.notna()
        valid_numeric = numeric.loc[valid_mask]

        if len(valid_numeric) < 4:
            return []

        q1 = valid_numeric.quantile(0.25)
        q3 = valid_numeric.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            return []

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (valid_numeric < lower_bound)
            | (valid_numeric > upper_bound)
        )

        outliers = valid_numeric.loc[outlier_mask]
        outlier_count = len(outliers)

        if outlier_count == 0:
            return []

        percentage = (
            outlier_count / len(valid_numeric) * 100
        )

        affected_positions = self._row_positions(
            numeric.notna()
            & (
                (numeric < lower_bound)
                | (numeric > upper_bound)
            )
        )

        return [
            self._create_issue(
                issue_type="potential_outliers",
                column_name=column_name,
                severity=self._severity(percentage),
                confidence=0.90,
                affected_records=outlier_count,
                evidence={
                    "method": "IQR",
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(
                        percentage,
                        2
                    ),
                    "sample_values": [
                        float(value)
                        for value in outliers.head(10)
                    ],
                    "affected_row_positions": affected_positions
                },
                suggested_action=(
                    "Investigate unusual values against business rules. "
                    "Do not remove valid extreme observations automatically."
                )
            )
        ]

    def _detect_mixed_types(
        self,
        series: pd.Series,
        column_name: str,
        semantic_type: str
    ) -> list[DataQualityIssue]:

        if semantic_type != "mixed":
            return []

        missing_mask = self._missing_mask(series)
        clean = series.loc[~missing_mask]

        numeric = self._to_numeric(clean)

        invalid_mask = numeric.isna()
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            return []

        percentage = (
            invalid_count / max(len(clean), 1) * 100
        )

        invalid_values = (
            clean.loc[invalid_mask]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        affected_positions = self._row_positions(
            ~missing_mask & numeric.isna()
        )

        return [
            self._create_issue(
                issue_type="mixed_data_types",
                column_name=column_name,
                severity=self._severity(percentage),
                confidence=0.95,
                affected_records=invalid_count,
                evidence={
                    "non_numeric_count": invalid_count,
                    "non_numeric_percentage": round(
                        percentage,
                        2
                    ),
                    "sample_non_numeric_values": invalid_values,
                    "affected_row_positions": affected_positions
                },
                suggested_action=(
                    "Inspect inconsistent values and determine whether "
                    "they should be converted, preserved or treated "
                    "as a separate category."
                )
            )
        ]

    @staticmethod
    def _is_categorical_candidate(
        series: pd.Series,
        semantic_type: str
    ) -> bool:

        if semantic_type == "categorical":
            return True

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        clean = series.dropna().astype("string").str.strip()
        clean = clean.loc[clean.ne("")]

        if len(clean) < 5:
            return False

        unique_count = int(clean.nunique())

        cardinality_limit = max(
            20,
            int(len(clean) * 0.05)
        )

        return (
            unique_count <= cardinality_limit
            and unique_count / len(clean) <= 0.20
        )

    def _detect_formatting(
        self,
        series: pd.Series,
        column_name: str,
        semantic_type: str
    ) -> list[DataQualityIssue]:

        if not self._is_categorical_candidate(
            series,
            semantic_type
        ):
            return []

        missing_mask = self._missing_mask(series)

        clean = series.loc[~missing_mask].astype(str)

        if clean.empty:
            return []

        normalized = clean.str.strip().str.casefold()

        groups = {}

        for original, normalized_value in zip(
            clean,
            normalized
        ):
            groups.setdefault(
                normalized_value,
                {}
            )

            groups[normalized_value][original] = (
                groups[normalized_value].get(original, 0) + 1
            )

        inconsistent_groups = {}
        affected_indices = []

        for normalized_value, variants in groups.items():
            if len(variants) <= 1:
                continue

            canonical_value = sorted(
                variants.items(),
                key=lambda item: (-item[1], item[0])
            )[0][0]

            inconsistent_variants = {
                value: count
                for value, count in variants.items()
                if value != canonical_value
            }

            if not inconsistent_variants:
                continue

            inconsistent_groups[normalized_value] = {
                "canonical_value": canonical_value,
                "variants": variants
            }

            affected_indices.extend(
                clean.index[clean.isin(inconsistent_variants)].tolist()
            )

        if not affected_indices:
            return []

        affected_mask = clean.index.isin(affected_indices)
        affected_count = int(affected_mask.sum())

        percentage = (
            affected_count / len(clean) * 100
        )

        row_positions = self._row_positions(
            series.index.to_series().isin(affected_indices)
        )

        return [
            self._create_issue(
                issue_type="inconsistent_categorical_formatting",
                column_name=column_name,
                severity=self._severity(percentage),
                confidence=0.95,
                affected_records=affected_count,
                evidence={
                    "variant_groups": {
                        key: details
                        for key, details in list(
                            inconsistent_groups.items()
                        )[:10]
                    },
                    "affected_records": affected_count,
                    "affected_percentage": round(
                        percentage,
                        2
                    ),
                    "affected_row_positions": row_positions
                },
                suggested_action=(
                    "Review equivalent category values and standardize "
                    "casing or whitespace where appropriate."
                )
            )
        ]

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:

        values = series.astype("string").str.strip()

        values = values.str.replace(
            ",",
            "",
            regex=False
        )

        return pd.to_numeric(
            values,
            errors="coerce"
        )
