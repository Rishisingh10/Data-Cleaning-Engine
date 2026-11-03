
import pandas as pd

from datadoctor.integrity import DataIntegrity


class CleaningValidator:

    def validate(
        self,
        original,
        cleaned,
        audit_log=None,
        expected_fingerprint=None
    ):
        audit_log = audit_log or []

        results = []

        def add_result(name, status, message, details=None):
            results.append({
                "check": name,
                "status": status,
                "message": message,
                "details": details or {}
            })

        if expected_fingerprint:
            try:
                DataIntegrity.validate_original(
                    original,
                    expected_fingerprint
                )
                add_result(
                    "Original data integrity",
                    "PASSED",
                    "Original dataset fingerprint is unchanged."
                )
            except Exception as error:
                add_result(
                    "Original data integrity",
                    "FAILED",
                    str(error)
                )

        if list(original.columns) == list(cleaned.columns):
            add_result(
                "Column structure",
                "PASSED",
                "Column names and order are unchanged."
            )
        else:
            add_result(
                "Column structure",
                "FAILED",
                "Column names or order changed unexpectedly."
            )

        original_dtypes = original.dtypes.astype(str).tolist()
        cleaned_dtypes = cleaned.dtypes.astype(str).tolist()

        if original_dtypes == cleaned_dtypes:
            add_result(
                "Data types",
                "PASSED",
                "All column data types are preserved."
            )
        else:
            add_result(
                "Data types",
                "FAILED",
                "One or more column data types changed.",
                {
                    "before": dict(zip(original.columns, original_dtypes)),
                    "after": dict(zip(cleaned.columns, cleaned_dtypes))
                }
            )

        row_difference = len(original) - len(cleaned)

        row_removal_recorded = any(
            entry.get("operation") in {
                "drop_missing_rows",
                "remove_exact_duplicates"
            }
            for entry in audit_log
        )

        if row_difference == 0:
            add_result(
                "Row count",
                "PASSED",
                "No rows were removed."
            )
        elif row_difference > 0 and row_removal_recorded:
            add_result(
                "Row count",
                "PASSED",
                f"{row_difference} rows were removed by recorded operations.",
                {
                    "rows_before": len(original),
                    "rows_after": len(cleaned),
                    "rows_removed": row_difference
                }
            )
        elif row_difference > 0:
            add_result(
                "Row count",
                "WARNING",
                f"{row_difference} rows were removed without a recognized row-removal audit entry."
            )
        else:
            add_result(
                "Row count",
                "WARNING",
                f"Cleaned dataset contains {-row_difference} additional rows."
            )

        if original.columns.equals(cleaned.columns):
            original_missing = int(original.isna().sum().sum())
            cleaned_missing = int(cleaned.isna().sum().sum())

            add_result(
                "Missing values",
                "PASSED" if cleaned_missing <= original_missing else "WARNING",
                "Missing values decreased or remained unchanged."
                if cleaned_missing <= original_missing
                else "Missing values increased after cleaning.",
                {
                    "before": original_missing,
                    "after": cleaned_missing,
                    "resolved": max(0, original_missing - cleaned_missing)
                }
            )

            original_duplicates = int(
                original.duplicated().sum()
            )

            cleaned_duplicates = int(
                cleaned.duplicated().sum()
            )

            add_result(
                "Duplicate rows",
                "PASSED" if cleaned_duplicates <= original_duplicates else "WARNING",
                "Duplicate rows decreased or remained unchanged."
                if cleaned_duplicates <= original_duplicates
                else "Duplicate rows increased after cleaning.",
                {
                    "before": original_duplicates,
                    "after": cleaned_duplicates
                }
            )

        try:
            cleaned_fingerprint = DataIntegrity.fingerprint(cleaned)

            add_result(
                "Cleaned dataset fingerprint",
                "PASSED",
                "Cleaned dataset fingerprint generated successfully.",
                {"fingerprint": cleaned_fingerprint}
            )
        except Exception as error:
            add_result(
                "Cleaned dataset fingerprint",
                "FAILED",
                str(error)
            )

        failed = sum(
            result["status"] == "FAILED"
            for result in results
        )

        warnings = sum(
            result["status"] == "WARNING"
            for result in results
        )

        return {
            "overall_status": (
                "FAILED" if failed
                else "WARNING" if warnings
                else "PASSED"
            ),
            "total_checks": len(results),
            "passed": sum(
                result["status"] == "PASSED"
                for result in results
            ),
            "warnings": warnings,
            "failed": failed,
            "checks": results
        }
