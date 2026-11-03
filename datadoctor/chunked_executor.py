
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path

import pandas as pd


class ChunkedCleaningExecutor:
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
        "review_outliers",
    }

    EXPECTED_ISSUE_TYPES = {
        "drop_missing_rows": {"missing_values"},
        "fill_mean": {"missing_values"},
        "fill_median": {"missing_values"},
        "fill_mode": {"missing_values"},
        "fill_constant": {"missing_values"},
        "normalize_whitespace": {
            "formatting_inconsistency",
            "inconsistent_categorical_formatting",
        },
        "standardize_case": {
            "formatting_inconsistency",
            "inconsistent_categorical_formatting",
        },
        "remove_exact_duplicates": {"duplicate_rows"},
        "review_outliers": {"outliers", "potential_outliers"},
    }

    def execute_csv(
        self,
        input_path,
        output_path,
        plan,
        approved_issue_ids,
        report,
        chunksize=50000,
        encoding="utf-8-sig",
        expected_fingerprint=None,
    ):
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path == output_path:
            raise ValueError("Input and output paths must be different.")

        if chunksize < 1:
            raise ValueError("Chunk size must be greater than zero.")

        if input_path.suffix.lower() != ".csv":
            raise ValueError("Chunked execution supports CSV files only.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        approved_issue_ids = set(approved_issue_ids or [])

        recommendations = [
            recommendation
            for recommendation in plan.recommendations
            if recommendation.issue_id in approved_issue_ids
        ]

        report_issues = {
            issue.issue_id: issue
            for issue in report.issues
        }

        for recommendation in recommendations:
            operation = recommendation.operation

            if operation not in self.SUPPORTED_OPERATIONS:
                raise ValueError(f"Unsupported operation: {operation}")

            if (
                report_issues
                and recommendation.issue_id not in report_issues
            ):
                raise ValueError(
                    f"Issue {recommendation.issue_id} does not exist in the report."
                )

            issue = report_issues.get(recommendation.issue_id)

            if issue and operation in self.EXPECTED_ISSUE_TYPES:
                allowed_types = self.EXPECTED_ISSUE_TYPES[operation]

                if issue.issue_type not in allowed_types:
                    raise ValueError(
                        f"Operation '{operation}' is not valid for "
                        f"issue type '{issue.issue_type}'."
                    )

            if operation in {
                "fill_mean",
                "fill_median",
                "fill_mode",
                "fill_constant",
                "normalize_whitespace",
                "standardize_case",
            }:
                if not recommendation.target_column:
                    raise ValueError(
                        f"Operation '{operation}' requires a target column."
                    )

        input_hash = self._file_hash(input_path)

        if (
            expected_fingerprint is not None
            and expected_fingerprint != input_hash
        ):
            raise ValueError(
                "Input fingerprint does not match the expected fingerprint."
            )

        fill_values = self._calculate_fill_values(
            input_path=input_path,
            recommendations=recommendations,
            chunksize=chunksize,
            encoding=encoding,
        )

        output_temp = None
        output_handle = None
        database_path = None
        connection = None

        rows_read = 0
        rows_written = 0
        duplicates_removed = 0
        missing_rows_removed = 0

        start_time = time.perf_counter()

        operations_applied = [
            {
                "issue_id": recommendation.issue_id,
                "operation": recommendation.operation,
                "target_column": recommendation.target_column,
                "parameters": recommendation.parameters,
            }
            for recommendation in recommendations
        ]

        try:
            output_handle = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix="datadoctor_",
                dir=output_path.parent,
                encoding=encoding,
                newline="",
                delete=False,
            )

            output_temp = Path(output_handle.name)

            database_handle = tempfile.NamedTemporaryFile(
                suffix=".sqlite",
                prefix="datadoctor_dedup_",
                delete=False,
            )

            database_path = Path(database_handle.name)
            database_handle.close()

            connection = sqlite3.connect(str(database_path))

            connection.execute(
                "CREATE TABLE seen_rows (row_hash TEXT PRIMARY KEY)"
            )

            reader = pd.read_csv(
                input_path,
                chunksize=chunksize,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )

            first_chunk = True

            for chunk in reader:
                rows_read += len(chunk)

                for recommendation in recommendations:
                    operation = recommendation.operation
                    column = recommendation.target_column
                    parameters = recommendation.parameters or {}

                    if operation in {
                        "fill_mean",
                        "fill_median",
                        "fill_mode",
                        "fill_constant",
                        "normalize_whitespace",
                        "standardize_case",
                    }:
                        if column not in chunk.columns:
                            raise ValueError(
                                f"Column '{column}' does not exist in the CSV."
                            )

                    if operation == "drop_missing_rows":
                        missing_mask = self._row_missing_mask(chunk)

                        removed = int(missing_mask.sum())
                        missing_rows_removed += removed

                        chunk = chunk.loc[~missing_mask].copy()

                    elif operation in {
                        "fill_mean",
                        "fill_median",
                        "fill_mode",
                        "fill_constant",
                    }:
                        missing_mask = self._missing_mask(chunk[column])

                        if operation == "fill_constant":
                            if "value" in parameters:
                                fill_value = parameters["value"]
                            elif "constant" in parameters:
                                fill_value = parameters["constant"]
                            else:
                                raise ValueError(
                                    "fill_constant requires a 'value' parameter."
                                )
                        else:
                            fill_value = fill_values.get(
                                (recommendation.issue_id, column)
                            )

                            if fill_value is None:
                                continue

                        chunk.loc[missing_mask, column] = str(fill_value)

                    elif operation == "normalize_whitespace":
                        chunk[column] = (
                            chunk[column]
                            .str.replace(r"\s+", " ", regex=True)
                            .str.strip()
                        )

                    elif operation == "standardize_case":
                        case = str(
                            parameters.get("case", "lower")
                        ).lower()

                        if case == "lower":
                            chunk[column] = chunk[column].str.lower()

                        elif case == "upper":
                            chunk[column] = chunk[column].str.upper()

                        elif case == "title":
                            chunk[column] = chunk[column].str.title()

                        else:
                            raise ValueError(
                                "standardize_case parameter 'case' must be "
                                "'lower', 'upper', or 'title'."
                            )

                if any(
                    recommendation.operation == "remove_exact_duplicates"
                    for recommendation in recommendations
                ):
                    row_hashes = chunk.apply(
                        self._row_hash,
                        axis=1,
                    )

                    keep_mask = []

                    for row_hash in row_hashes:
                        cursor = connection.execute(
                            "INSERT OR IGNORE INTO seen_rows (row_hash) VALUES (?)",
                            (row_hash,),
                        )

                        keep_mask.append(cursor.rowcount == 1)

                    duplicates_removed += (
                        len(keep_mask) - sum(keep_mask)
                    )

                    chunk = chunk.loc[keep_mask].copy()

                chunk.to_csv(
                    output_handle,
                    header=first_chunk,
                    index=False,
                    lineterminator="\n",
                    na_rep="",
                )

                output_handle.flush()

                rows_written += len(chunk)
                first_chunk = False

            if first_chunk:
                empty_frame = pd.read_csv(
                    input_path,
                    nrows=0,
                    encoding=encoding,
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                )

                empty_frame.to_csv(
                    output_handle,
                    index=False,
                    lineterminator="\n",
                )

            output_handle.close()
            output_handle = None

            connection.commit()
            connection.close()
            connection = None

            output_hash = self._file_hash(output_temp)
            final_input_hash = self._file_hash(input_path)

            if input_hash != final_input_hash:
                raise RuntimeError(
                    "Input file changed during execution. "
                    "Output was not committed."
                )

            os.replace(output_temp, output_path)
            output_temp = None

            elapsed_time = time.perf_counter() - start_time

            return {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "rows_read": rows_read,
                "rows_written": rows_written,
                "duplicates_removed": duplicates_removed,
                "missing_rows_removed": missing_rows_removed,
                "operations": operations_applied,
                "execution_time_seconds": round(elapsed_time, 3),
                "rows_per_second": round(
                    rows_read / elapsed_time if elapsed_time else 0,
                    2,
                ),
                "input_sha256": input_hash,
                "output_sha256": output_hash,
            }

        except Exception:
            if output_handle is not None:
                output_handle.close()

            if connection is not None:
                connection.close()

            if output_temp is not None and output_temp.exists():
                output_temp.unlink()

            raise

        finally:
            if database_path is not None and database_path.exists():
                database_path.unlink()

    def _calculate_fill_values(
        self,
        input_path,
        recommendations,
        chunksize,
        encoding,
    ):
        required = [
            recommendation
            for recommendation in recommendations
            if recommendation.operation in {
                "fill_mean",
                "fill_median",
                "fill_mode",
            }
        ]

        if not required:
            return {}

        statistics = {}

        for recommendation in required:
            key = (
                recommendation.issue_id,
                recommendation.target_column,
            )

            statistics[key] = {
                "operation": recommendation.operation,
                "numeric_values": [],
                "sum": 0.0,
                "count": 0,
                "frequencies": Counter(),
            }

        columns = list({
            recommendation.target_column
            for recommendation in required
        })

        reader = pd.read_csv(
            input_path,
            chunksize=chunksize,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            usecols=lambda column: column in columns,
        )

        for chunk in reader:
            for recommendation in required:
                column = recommendation.target_column

                if column not in chunk.columns:
                    raise ValueError(
                        f"Column '{column}' does not exist in the CSV."
                    )

                key = (
                    recommendation.issue_id,
                    column,
                )

                operation = recommendation.operation
                values = chunk[column]

                valid_values = values[
                    ~self._missing_mask(values)
                ]

                if operation in {"fill_mean", "fill_median"}:
                    numeric_values = pd.to_numeric(
                        valid_values,
                        errors="coerce",
                    ).dropna()

                    if operation == "fill_mean":
                        statistics[key]["sum"] += float(
                            numeric_values.sum()
                        )

                        statistics[key]["count"] += len(numeric_values)

                    else:
                        statistics[key]["numeric_values"].extend(
                            numeric_values.tolist()
                        )

                elif operation == "fill_mode":
                    statistics[key]["frequencies"].update(
                        valid_values.tolist()
                    )

        results = {}

        for key, values in statistics.items():
            operation = values["operation"]

            if operation == "fill_mean":
                count = values["count"]

                if count:
                    results[key] = values["sum"] / count

            elif operation == "fill_median":
                numeric_values = values["numeric_values"]

                if numeric_values:
                    results[key] = float(
                        pd.Series(numeric_values).median()
                    )

            elif operation == "fill_mode":
                frequencies = values["frequencies"]

                if frequencies:
                    results[key] = frequencies.most_common(1)[0][0]

        return results

    @staticmethod
    def _missing_mask(series):
        return (
            series.isna()
            | series.astype(str).str.strip().eq("")
        )

    @classmethod
    def _row_missing_mask(cls, dataframe):
        return dataframe.apply(
            lambda column: cls._missing_mask(column)
        ).any(axis=1)

    @classmethod
    def _row_hash(cls, row):
        serialized = json.dumps(
            [
                None if pd.isna(value) else str(value)
                for value in row
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _file_hash(file_path):
        digest = hashlib.sha256()

        with open(file_path, "rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(block)

        return digest.hexdigest()
