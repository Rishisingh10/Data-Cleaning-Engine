
import re
from collections import Counter
from itertools import combinations

import pandas as pd


class AdvancedDataIntelligence:

    EMAIL_PATTERN = re.compile(
        r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    )

    URL_PATTERN = re.compile(
        r"^https?://[^\s]+$",
        re.IGNORECASE
    )

    def analyze(self, dataframe: pd.DataFrame) -> dict:
        return {
            "text_analysis": self._analyze_text_columns(dataframe),
            "semantic_candidates": self._infer_semantic_candidates(dataframe),
            "anomalies": self._detect_anomalies(dataframe),
            "cross_column_checks": self._cross_column_checks(dataframe)
        }

    def _analyze_text_columns(self, dataframe):
        results = {}

        for column in dataframe.columns:
            series = dataframe[column]

            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                continue

            values = series.dropna().astype(str)
            non_empty = values[values.str.strip() != ""]

            lengths = non_empty.str.len()
            words = non_empty.str.split().str.len()

            frequencies = Counter(non_empty.tolist())

            email_valid = non_empty.map(
                lambda value: bool(
                    self.EMAIL_PATTERN.fullmatch(value.strip())
                )
            )

            url_valid = non_empty.map(
                lambda value: bool(
                    self.URL_PATTERN.fullmatch(value.strip())
                )
            )

            email_candidates = non_empty[
                non_empty.str.contains("@", regex=False)
            ]

            url_candidates = non_empty[
                non_empty.str.contains(
                    r"https?://",
                    case=False,
                    regex=True
                )
            ]

            results[str(column)] = {
                "non_null_count": int(len(values)),
                "empty_string_count": int(
                    (values.str.strip() == "").sum()
                ),
                "unique_count": int(non_empty.nunique()),
                "duplicate_value_count": int(
                    len(non_empty) - non_empty.nunique()
                ),
                "character_length": {
                    "minimum": int(lengths.min()) if len(lengths) else 0,
                    "maximum": int(lengths.max()) if len(lengths) else 0,
                    "average": round(float(lengths.mean()), 2) if len(lengths) else 0
                },
                "word_count": {
                    "minimum": int(words.min()) if len(words) else 0,
                    "maximum": int(words.max()) if len(words) else 0,
                    "average": round(float(words.mean()), 2) if len(words) else 0
                },
                "leading_trailing_spaces": int(
                    (non_empty != non_empty.str.strip()).sum()
                ),
                "repeated_whitespace": int(
                    non_empty.str.contains(r"\s{2,}", regex=True).sum()
                ),
                "email_candidates": int(len(email_candidates)),
                "valid_email_count": int(
                    email_candidates.map(
                        lambda value: bool(
                            self.EMAIL_PATTERN.fullmatch(value.strip())
                        )
                    ).sum()
                ),
                "invalid_email_candidates": int(
                    email_candidates.map(
                        lambda value: not bool(
                            self.EMAIL_PATTERN.fullmatch(value.strip())
                        )
                    ).sum()
                ),
                "url_candidates": int(len(url_candidates)),
                "valid_url_count": int(
                    url_candidates.map(
                        lambda value: bool(
                            self.URL_PATTERN.fullmatch(value.strip())
                        )
                    ).sum()
                ),
                "invalid_url_candidates": int(
                    url_candidates.map(
                        lambda value: not bool(
                            self.URL_PATTERN.fullmatch(value.strip())
                        )
                    ).sum()
                ),
                "most_common_values": [
                    {"value": str(value), "count": int(count)}
                    for value, count in frequencies.most_common(5)
                ]
            }

        return results

    def _infer_semantic_candidates(self, dataframe):
        results = {}

        for column in dataframe.columns:
            series = dataframe[column]
            values = series.dropna().astype(str).str.strip()

            if values.empty:
                continue

            sample = values.head(500)
            name = str(column).lower()

            email_matches = sample.map(
                lambda value: bool(
                    self.EMAIL_PATTERN.fullmatch(value)
                )
            )

            url_matches = sample.map(
                lambda value: bool(
                    self.URL_PATTERN.fullmatch(value)
                )
            )

            email_ratio = email_matches.mean()
            url_ratio = url_matches.mean()

            numeric_ratio = pd.to_numeric(
                sample,
                errors="coerce"
            ).notna().mean()

            date_ratio = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed"
            ).notna().mean()

            email_column_hint = any(
                term in name
                for term in ["email", "e-mail", "mail_address"]
            )

            url_column_hint = any(
                term in name
                for term in ["url", "website", "web_address", "link"]
            )

            identifier_hint = any(
                term in name
                for term in ["id", "code", "key"]
            )

            if email_ratio >= 0.6 and email_matches.any():
                candidate = "email"
            elif email_column_hint and email_matches.any():
                candidate = "email"
            elif url_ratio >= 0.8 and url_matches.any():
                candidate = "url"
            elif url_column_hint and url_matches.any():
                candidate = "url"
            elif identifier_hint:
                candidate = "identifier_candidate"
            elif date_ratio >= 0.8 and numeric_ratio < 0.8:
                candidate = "datetime"
            elif numeric_ratio >= 0.8:
                candidate = "numerical_text"
            else:
                candidate = "text_or_categorical"

            results[str(column)] = {
                "candidate_type": candidate,
                "sample_size": int(len(sample)),
                "email_match_ratio": round(float(email_ratio), 3),
                "url_match_ratio": round(float(url_ratio), 3),
                "numeric_match_ratio": round(float(numeric_ratio), 3),
                "date_match_ratio": round(float(date_ratio), 3)
            }

        return results

    def _detect_anomalies(self, dataframe):
        results = []

        for column in dataframe.columns:
            series = dataframe[column]

            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.dropna()

            if len(valid) >= 5 and valid.nunique() > 1:
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1

                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr

                    mask = (valid < lower) | (valid > upper)

                    if mask.any():
                        results.append({
                            "column": str(column),
                            "anomaly_type": "statistical_outlier",
                            "affected_records": int(mask.sum()),
                            "lower_bound": float(lower),
                            "upper_bound": float(upper),
                            "sample_values": ", ".join(
                                str(value)
                                for value in valid[mask].head(5).tolist()
                            ),
                            "requires_review": True
                        })

            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                values = series.dropna().astype(str).str.strip()
                counts = values.value_counts()

                if len(values) >= 10 and len(counts) > 1:
                    rare = counts[
                        counts <= max(1, int(len(values) * 0.01))
                    ]

                    if not rare.empty:
                        results.append({
                            "column": str(column),
                            "anomaly_type": "rare_category",
                            "affected_records": int(rare.sum()),
                            "sample_values": ", ".join(
                                str(value)
                                for value in rare.index[:5].tolist()
                            ),
                            "requires_review": True
                        })

        return results

    def _cross_column_checks(self, dataframe):
        results = []

        columns = list(dataframe.columns)

        for first, second in combinations(columns, 2):
            if dataframe[first].equals(dataframe[second]):
                results.append({
                    "check_type": "identical_columns",
                    "columns": [str(first), str(second)],
                    "message": "These columns contain identical values.",
                    "requires_review": True
                })

        numeric = dataframe.select_dtypes(include="number")

        if numeric.shape[1] >= 2:
            correlation = numeric.corr(min_periods=5)

            for first, second in combinations(correlation.columns, 2):
                value = correlation.loc[first, second]

                if pd.notna(value) and abs(value) >= 0.98:
                    results.append({
                        "check_type": "high_correlation",
                        "columns": [str(first), str(second)],
                        "correlation": round(float(value), 4),
                        "message": (
                            "These numerical columns are highly correlated. "
                            "Review whether they represent overlapping information."
                        ),
                        "requires_review": True
                    })

        return results
