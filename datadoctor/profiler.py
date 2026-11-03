
import re
from typing import Any

import numpy as np
import pandas as pd

from datadoctor.schemas import ColumnProfile, DatasetProfile


class DataProfiler:

    ID_PATTERN = re.compile(
        r"(^id$|[\s_-]id$|^id[\s_-]|customer.?id|user.?id|"
        r"account.?id|_id$|identifier|uuid|primary.?key|"
        r"customer.?number|account.?number|record.?number)",
        re.IGNORECASE
    )

    CURRENCY_PATTERN = re.compile(
        r"(amount|price|salary|revenue|income|cost|spend|"
        r"payment|balance|usd|currency)",
        re.IGNORECASE
    )

    DATE_PATTERN = re.compile(
        r"(date|datetime|timestamp|dob|birth)",
        re.IGNORECASE
    )

    LOCATION_PATTERN = re.compile(
        r"(location|city|country|state|address|region|"
        r"latitude|longitude|postal)",
        re.IGNORECASE
    )

    CATEGORY_PATTERN = re.compile(
        r"(gender|category|type|status|season|color|size|"
        r"department|grade|rating|class)",
        re.IGNORECASE
    )

    EMAIL_PATTERN = re.compile(
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )

    URL_PATTERN = re.compile(
        r"^(https?://|www\.)\S+$",
        re.IGNORECASE
    )

    BOOLEAN_VALUES = {
        "true", "false", "yes", "no", "y", "n"
    }

    def profile_dataframe(
        self,
        dataframe: pd.DataFrame,
        file_name: str = "dataset",
        sheet_name: str | None = None
    ) -> DatasetProfile:

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        profiles = {}

        for index, column in enumerate(dataframe.columns):
            name = str(column)

            key = name if name not in profiles else f"{name}_{index + 1}"

            profiles[key] = self._profile_column(
                column_name=name,
                series=dataframe.iloc[:, index]
            )

        return DatasetProfile(
            file_name=file_name,
            sheet_name=sheet_name,
            total_records=len(dataframe),
            total_columns=len(dataframe.columns),
            column_profiles=profiles
        )

    def _profile_column(
        self,
        column_name: str,
        series: pd.Series
    ) -> ColumnProfile:

        missing_mask = series.isna()

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            blank_mask = (
                series.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )
            missing_mask |= blank_mask

        clean = series.loc[~missing_mask]

        total_records = len(series)
        missing_count = int(missing_mask.sum())
        non_null_count = len(clean)
        unique_count = int(clean.nunique())

        semantic_type, confidence, evidence = (
            self._infer_semantic_type(
                column_name,
                series,
                clean
            )
        )

        statistics = self._calculate_statistics(
            clean,
            semantic_type
        )

        warnings = []

        if missing_count:
            warnings.append(
                f"{missing_count} records contain missing or blank values."
            )

        if semantic_type == "mixed":
            warnings.append(
                "The column contains values with inconsistent data types."
            )

        if semantic_type == "identifier":
            warnings.append(
                "Statistical measurements are not applicable to identifiers."
            )

        if semantic_type == "identifier_candidate":
            warnings.append(
                "This column may contain identifiers. Verify before applying numerical operations."
            )

        return ColumnProfile(
            column_name=column_name,
            physical_type=str(series.dtype),
            semantic_type=semantic_type,
            semantic_confidence=confidence,
            inference_evidence=evidence,
            total_records=total_records,
            non_null_count=non_null_count,
            missing_count=missing_count,
            unique_count=unique_count,
            statistics=statistics,
            sample_values=self._safe_samples(clean),
            warnings=warnings
        )

    def _infer_semantic_type(
        self,
        column_name: str,
        original: pd.Series,
        clean: pd.Series
    ) -> tuple[str, float, list[str]]:

        if clean.empty:
            return "empty", 1.0, ["No non-missing values found."]

        name = column_name.strip().lower()
        values = clean.astype("string").str.strip()
        lower_values = values.str.lower()

        if self.ID_PATTERN.search(name):
            return (
                "identifier",
                0.99,
                ["Column name matches an identifier pattern."]
            )

        if pd.api.types.is_bool_dtype(original):
            return (
                "boolean",
                1.0,
                ["Column has a native boolean data type."]
            )

        if lower_values.isin(
            self.BOOLEAN_VALUES | {"1", "0"}
        ).all() and lower_values.nunique() <= 2:
            return (
                "boolean",
                0.95,
                ["Values match common boolean representations."]
            )

        if pd.api.types.is_datetime64_any_dtype(original):
            return (
                "datetime",
                1.0,
                ["Pandas identifies the column as datetime."]
            )

        if self.DATE_PATTERN.search(name):
            parsed = pd.to_datetime(clean, errors="coerce")
            valid_ratio = float(parsed.notna().mean())

            if valid_ratio >= 0.8:
                return (
                    "datetime",
                    valid_ratio,
                    ["Column name and values indicate datetime data."]
                )

        date_like = values.str.match(
            r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$"
        )

        if date_like.mean() >= 0.8:
            parsed = pd.to_datetime(clean, errors="coerce")
            valid_ratio = float(parsed.notna().mean())

            if valid_ratio >= 0.8:
                return (
                    "datetime",
                    valid_ratio,
                    ["Most values match recognizable date patterns."]
                )

        email_ratio = lower_values.map(
            lambda value: bool(self.EMAIL_PATTERN.match(value))
        ).mean()

        if email_ratio >= 0.8:
            return (
                "email",
                float(email_ratio),
                ["Most values match an email address pattern."]
            )

        url_ratio = values.map(
            lambda value: bool(self.URL_PATTERN.match(value))
        ).mean()

        if url_ratio >= 0.8:
            return (
                "url",
                float(url_ratio),
                ["Most values match a URL pattern."]
            )

        if self.LOCATION_PATTERN.search(name):
            return (
                "location",
                0.90,
                ["Column name indicates geographic information."]
            )

        numeric = self._to_numeric(clean)
        numeric_ratio = float(numeric.notna().mean())

        if self.CURRENCY_PATTERN.search(name) and numeric_ratio >= 0.8:
            return (
                "currency",
                numeric_ratio,
                ["Column name indicates monetary data and values are numeric."]
            )

        if numeric_ratio == 1.0:

            if pd.api.types.is_numeric_dtype(original):
                if (
                    len(clean) >= 20
                    and clean.nunique() / len(clean) >= 0.98
                    and pd.api.types.is_integer_dtype(original)
                ):
                    return (
                        "identifier_candidate",
                        0.75,
                        ["Integer column has very high cardinality."]
                    )

                return (
                    "numerical",
                    0.99,
                    ["All non-missing values are numeric."]
                )

            return (
                "numerical_text",
                0.95,
                ["All non-missing values can be converted to numbers."]
            )

        if numeric_ratio >= 0.8:
            return (
                "mixed",
                numeric_ratio,
                [
                    f"{numeric_ratio:.1%} of values are numeric.",
                    "Remaining values could not be converted to numbers."
                ]
            )

        if self.CATEGORY_PATTERN.search(name):
            return (
                "categorical",
                0.95,
                ["Column name indicates categorical information."]
            )

        unique_ratio = clean.nunique() / max(len(clean), 1)
        average_length = values.str.len().mean()

        if (
            len(clean) >= 10
            and unique_ratio <= 0.2
            and clean.nunique() <= 20
            and average_length <= 60
        ):
            return (
                "categorical",
                0.90,
                ["Low cardinality relative to the number of records."]
            )

        return (
            "text",
            0.85,
            ["Values appear to contain free-form or high-cardinality text."]
        )

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:

        values = series.astype("string").str.strip()

        values = values.str.replace(",", "", regex=False)

        return pd.to_numeric(
            values,
            errors="coerce"
        )

    def _calculate_statistics(
        self,
        series: pd.Series,
        semantic_type: str
    ) -> dict[str, Any]:

        if series.empty:
            return {}

        if semantic_type in {
            "identifier",
            "identifier_candidate",
            "empty"
        }:
            return {
                "distinct_values": int(series.nunique())
            }

        if semantic_type in {
            "numerical",
            "numerical_text",
            "currency"
        }:
            numeric = self._to_numeric(series).dropna()

            if numeric.empty:
                return {}

            q1 = numeric.quantile(0.25)
            median = numeric.median()
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = int(
                (
                    (numeric < lower_bound)
                    | (numeric > upper_bound)
                ).sum()
            )

            modes = numeric.mode().head(3).tolist()

            return {
                "count": int(numeric.count()),
                "mean": float(numeric.mean()),
                "median": float(median),
                "mode": [float(value) for value in modes],
                "standard_deviation": self._safe_number(numeric.std()),
                "variance": self._safe_number(numeric.var()),
                "minimum": float(numeric.min()),
                "maximum": float(numeric.max()),
                "range": float(numeric.max() - numeric.min()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "outlier_count_iqr": outlier_count,
                "outlier_percentage_iqr": round(
                    outlier_count / len(numeric) * 100,
                    2
                ),
                "skewness": self._safe_number(numeric.skew()),
                "kurtosis": self._safe_number(numeric.kurt())
            }

        if semantic_type in {
            "categorical",
            "boolean",
            "location"
        }:
            frequencies = series.astype(str).value_counts()

            return {
                "count": int(len(series)),
                "unique_values": int(series.nunique()),
                "cardinality_ratio": round(
                    series.nunique() / len(series),
                    4
                ),
                "top_values": [
                    {
                        "value": str(value),
                        "count": int(count),
                        "percentage": round(
                            count / len(series) * 100,
                            2
                        )
                    }
                    for value, count in frequencies.head(10).items()
                ]
            }

        if semantic_type == "datetime":
            dates = pd.to_datetime(
                series,
                errors="coerce"
            ).dropna()

            if dates.empty:
                return {}

            return {
                "valid_dates": int(len(dates)),
                "minimum_date": str(dates.min()),
                "maximum_date": str(dates.max()),
                "year_distribution": {
                    str(year): int(count)
                    for year, count in dates.dt.year.value_counts().items()
                },
                "month_distribution": {
                    str(month): int(count)
                    for month, count in dates.dt.month.value_counts().items()
                }
            }

        if semantic_type in {
            "text",
            "email",
            "url",
            "mixed"
        }:
            values = series.astype("string")

            lengths = values.str.len()
            word_counts = values.str.split().str.len()

            return {
                "count": int(len(values)),
                "average_characters": float(lengths.mean()),
                "minimum_characters": int(lengths.min()),
                "maximum_characters": int(lengths.max()),
                "average_words": float(word_counts.mean()),
                "minimum_words": int(word_counts.min()),
                "maximum_words": int(word_counts.max()),
                "whitespace_issues": int(
                    values.ne(values.str.strip()).sum()
                ),
                "empty_strings": int(
                    values.str.strip().eq("").sum()
                ),
                "special_character_values": int(
                    values.str.contains(
                        r"[^a-zA-Z0-9\s]",
                        regex=True,
                        na=False
                    ).sum()
                )
            }

        return {
            "count": int(len(series)),
            "unique_values": int(series.nunique())
        }

    @staticmethod
    def _safe_number(value: Any) -> float | None:

        if pd.isna(value) or not np.isfinite(value):
            return None

        return float(value)

    @staticmethod
    def _safe_samples(
        series: pd.Series,
        limit: int = 5
    ) -> list[Any]:

        samples = series.dropna().head(limit).tolist()
        result = []

        for value in samples:
            if isinstance(value, np.generic):
                value = value.item()

            if isinstance(value, (pd.Timestamp,)):
                value = value.isoformat()

            result.append(value)

        return result
