import io

import pandas as pd


class QualityReportGenerator:

    def _to_dict(self, value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")

        if hasattr(value, "dict"):
            return value.dict()

        if isinstance(value, dict):
            return value

        return {}

    def _get_columns(self, profile):
        profile_data = self._to_dict(profile)
        columns = profile_data.get("columns", [])

        if isinstance(columns, dict):
            return [
                {"column_name": name, **details}
                for name, details in columns.items()
            ]

        return columns

    def _get_issues(self, report):
        report_data = self._to_dict(report)
        return [
            self._to_dict(issue)
            for issue in report_data.get("issues", [])
        ]

    def generate_excel(
        self,
        original,
        cleaned,
        original_profile,
        cleaned_profile,
        quality_report,
        audit_log,
        validation,
        business_rule_results=None
    ):
        buffer = io.BytesIO()

        before_missing = int(original.isna().sum().sum())
        after_missing = int(cleaned.isna().sum().sum())

        before_duplicates = int(original.duplicated().sum())
        after_duplicates = int(cleaned.duplicated().sum())

        summary = pd.DataFrame([
            {"Metric": "Original rows", "Value": len(original)},
            {"Metric": "Cleaned rows", "Value": len(cleaned)},
            {"Metric": "Original columns", "Value": len(original.columns)},
            {"Metric": "Cleaned columns", "Value": len(cleaned.columns)},
            {"Metric": "Missing cells before", "Value": before_missing},
            {"Metric": "Missing cells after", "Value": after_missing},
            {"Metric": "Missing cells resolved", "Value": max(0, before_missing - after_missing)},
            {"Metric": "Duplicate rows before", "Value": before_duplicates},
            {"Metric": "Duplicate rows after", "Value": after_duplicates},
            {"Metric": "Validation status", "Value": validation.get("overall_status", "UNKNOWN")},
            {"Metric": "Validation checks passed", "Value": validation.get("passed", 0)},
            {"Metric": "Validation warnings", "Value": validation.get("warnings", 0)},
            {"Metric": "Validation failures", "Value": validation.get("failed", 0)}
        ])

        before_columns = self._get_columns(original_profile)
        after_columns = self._get_columns(cleaned_profile)

        before_map = {
            column.get("column_name", column.get("name")): column
            for column in before_columns
        }

        after_map = {
            column.get("column_name", column.get("name")): column
            for column in after_columns
        }

        column_rows = []

        for column in original.columns:
            name = str(column)
            before = before_map.get(name, {})
            after = after_map.get(name, {})

            column_rows.append({
                "Column": name,
                "Data Type Before": str(original[column].dtype),
                "Data Type After": str(cleaned[column].dtype) if column in cleaned.columns else "Missing",
                "Missing Before": int(original[column].isna().sum()),
                "Missing After": int(cleaned[column].isna().sum()) if column in cleaned.columns else None,
                "Unique Before": int(original[column].nunique(dropna=True)),
                "Unique After": int(cleaned[column].nunique(dropna=True)) if column in cleaned.columns else None,
                "Semantic Type Before": before.get("semantic_type", "Unknown"),
                "Semantic Type After": after.get("semantic_type", "Unknown")
            })

        issues = self._get_issues(quality_report)

        validation_rows = validation.get("checks", [])

        audit_rows = [
            self._to_dict(entry)
            for entry in audit_log
        ]

        business_rule_results = business_rule_results or {}
        business_rule_rows = business_rule_results.get("results", [])
        business_rule_summary = pd.DataFrame([
            {"Metric": "Total rules", "Value": business_rule_results.get("total_rules", 0)},
            {"Metric": "Rules passed", "Value": business_rule_results.get("rules_passed", 0)},
            {"Metric": "Rules failed", "Value": business_rule_results.get("rules_failed", 0)},
            {"Metric": "Total violations", "Value": business_rule_results.get("total_violations", 0)}
        ])

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

            pd.DataFrame(column_rows).to_excel(
                writer,
                sheet_name="Column Comparison",
                index=False
            )

            pd.DataFrame(issues).to_excel(
                writer,
                sheet_name="Detected Issues",
                index=False
            )

            pd.DataFrame(audit_rows).to_excel(
                writer,
                sheet_name="Audit Log",
                index=False
            )

            pd.DataFrame(validation_rows).to_excel(
                writer,
                sheet_name="Validation",
                index=False
            )

            business_rule_summary.to_excel(
                writer,
                sheet_name="Business Rule Summary",
                index=False
            )

            pd.DataFrame(business_rule_rows).to_excel(
                writer,
                sheet_name="Business Rule Results",
                index=False
            )

        buffer.seek(0)
        return buffer.getvalue()