import hashlib
import json
import os
import tempfile

import pandas as pd
import streamlit as st

from datadoctor.audit import AuditManager
from datadoctor.advanced_intelligence import AdvancedDataIntelligence
from datadoctor.business_rules import BusinessRuleEngine
from datadoctor.chunked_executor import ChunkedCleaningExecutor
from datadoctor.cleaning_executor import CleaningExecutor
from datadoctor.cleaning_history import CleaningHistory
from datadoctor.detectors import DataQualityDetector
from datadoctor.integrity import DataIntegrity
from datadoctor.loader import DatasetLoader
from datadoctor.performance import DatasetPerformance
from datadoctor.privacy import PrivacyManager
from datadoctor.profiler import DataProfiler
from datadoctor.report_generator import QualityReportGenerator
from datadoctor.rule_builder import BusinessRuleBuilder
from datadoctor.schema_drift import SchemaDriftDetector
from datadoctor.validation import CleaningValidator


st.set_page_config(
    page_title="Data Cleaning ",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_services():
    return {
        "loader": DatasetLoader(),
        "profiler": DataProfiler(),
        "detector": DataQualityDetector(),
        "executor": CleaningExecutor(),
        "history": CleaningHistory(),
        "advanced": AdvancedDataIntelligence(),
        "validator": CleaningValidator(),
        "report_generator": QualityReportGenerator(),
        "business_rules": BusinessRuleEngine(),
        "rule_builder": BusinessRuleBuilder(),
        "privacy": PrivacyManager(),
        "performance": DatasetPerformance(),
        "audit": AuditManager(),
        "schema_drift": SchemaDriftDetector(),
        "chunked_executor": ChunkedCleaningExecutor()
    }


def initialize_state():
    defaults = {
        "original_dataframe": None,
        "cleaned_dataframe": None,
        "original_profile": None,
        "cleaned_profile": None,
        "quality_report": None,
        "cleaning_plan": None,
        "approved_issue_ids": [],
        "audit_log": [],
        "validation_results": None,
        "business_rule_results": None,
        "dataset_fingerprint": None,
        "dataset_filename": None,
        "selected_sheet": None,
        "analysis_complete": False,
        "cleaning_complete": False,
        "configured_business_rules": [],
        "privacy_findings": [],
        "gemini_redacted_columns": [],
        "advanced_results": None,
        "excel_report": None,
        "last_uploaded_signature": None,
        "source_temp_path": None,
        "chunk_output_path": None,
        "chunked_audit": [],
        "chunked_mode": False,
        "schema_drift_result": None,
        "schema_drift_signature": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def to_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "dict"):
        return value.dict()

    return value if isinstance(value, dict) else {}


def get_issues(report):
    return [to_dict(issue) for issue in report.issues]


def get_recommendations(plan):
    return [to_dict(item) for item in plan.recommendations]


def make_arrow_safe(dataframe):
    safe = dataframe.copy()

    for column in safe.columns:
        series = safe[column]

        if not pd.api.types.is_object_dtype(series.dtype):
            continue

        values = series.dropna()

        if values.empty:
            continue

        value_types = {type(value) for value in values}

        contains_nested_values = any(
            isinstance(value, (dict, list, tuple, set))
            for value in values
        )

        if len(value_types) > 1 or contains_nested_values:
            def format_value(value):
                if value is None or value is pd.NA or value is pd.NaT:
                    return None

                if isinstance(value, (dict, list, tuple, set)):
                    try:
                        return json.dumps(
                            value,
                            default=str,
                            ensure_ascii=False
                        )
                    except (TypeError, ValueError):
                        return str(value)

                try:
                    if pd.isna(value):
                        return None
                except (TypeError, ValueError):
                    pass

                return str(value)

            safe[column] = series.map(format_value)

    return safe


def display_dataframe(dataframe, **kwargs):
    st.dataframe(
        make_arrow_safe(dataframe),
        **kwargs
    )


def remove_temp_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def reset_analysis():
    remove_temp_file(st.session_state.get("source_temp_path"))
    remove_temp_file(st.session_state.get("chunk_output_path"))

    keys = [
        "original_dataframe",
        "cleaned_dataframe",
        "original_profile",
        "cleaned_profile",
        "quality_report",
        "cleaning_plan",
        "approved_issue_ids",
        "audit_log",
        "validation_results",
        "business_rule_results",
        "dataset_fingerprint",
        "dataset_filename",
        "selected_sheet",
        "analysis_complete",
        "cleaning_complete",
        "configured_business_rules",
        "privacy_findings",
        "gemini_redacted_columns",
        "advanced_results",
        "excel_report",
        "source_temp_path",
        "chunk_output_path",
        "chunked_audit",
        "chunked_mode",
        "schema_drift_result",
        "schema_drift_signature"
    ]

    for key in keys:
        st.session_state.pop(key, None)

    st.session_state.last_uploaded_signature = None
    initialize_state()
    st.rerun()


def load_dataset(uploaded_file, services):
    uploaded_file.seek(0)

    return services["loader"].load(
        uploaded_file,
        file_name=uploaded_file.name
    )


def save_uploaded_source(uploaded_file):
    remove_temp_file(st.session_state.get("source_temp_path"))

    suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        delete=False
    ) as temporary_file:
        temporary_file.write(uploaded_file.getvalue())
        path = temporary_file.name

    st.session_state.source_temp_path = path
    return path


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def count_csv_rows(path, chunksize=50000):
    total = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += len(chunk)

    return total


def analyze_dataset(dataframe, filename, sheet_name, services):
    if dataframe.empty:
        raise ValueError("The selected dataset contains no rows.")

    if len(dataframe.columns) == 0:
        raise ValueError("The selected dataset contains no columns.")

    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names are not supported: {duplicates}"
        )

    profile = services["profiler"].profile_dataframe(
        dataframe,
        file_name=filename,
        sheet_name=sheet_name
    )

    report = services["detector"].detect(
        dataframe,
        profile
    )

    st.session_state.original_dataframe = dataframe.copy(deep=True)
    st.session_state.original_profile = profile
    st.session_state.quality_report = report

    st.session_state.dataset_fingerprint = DataIntegrity.fingerprint(
        dataframe
    )

    st.session_state.dataset_filename = filename
    st.session_state.selected_sheet = sheet_name
    st.session_state.analysis_complete = True

    st.session_state.cleaned_dataframe = None
    st.session_state.cleaned_profile = None
    st.session_state.cleaning_plan = None
    st.session_state.approved_issue_ids = []
    st.session_state.audit_log = []
    st.session_state.validation_results = None
    st.session_state.business_rule_results = None
    st.session_state.configured_business_rules = []
    st.session_state.privacy_findings = []
    st.session_state.gemini_redacted_columns = []
    st.session_state.advanced_results = None
    st.session_state.excel_report = None
    st.session_state.cleaning_complete = False
    st.session_state.chunked_audit = []
    st.session_state.chunk_output_path = None


def display_dataset_overview(dataframe, profile, services):
    st.subheader("Dataset Overview")

    performance = services["performance"].assess(dataframe)

    columns = st.columns(4)

    columns[0].metric("Rows", f"{len(dataframe):,}")
    columns[1].metric("Columns", f"{len(dataframe.columns):,}")
    columns[2].metric(
        "Missing Cells",
        f"{int(dataframe.isna().sum().sum()):,}"
    )
    columns[3].metric(
        "Duplicate Rows",
        f"{int(dataframe.duplicated().sum()):,}"
    )

    st.caption(
        f"Memory usage: {performance['memory_mb']} MB · "
        f"Processing tier: {performance['tier'].title()} · "
        f"Suggested sample size: "
        f"{performance['recommended_sample_size']:,}"
    )

    profile_rows = []

    for column in profile.column_profiles.values():
        profile_rows.append({
            "Column": column.column_name,
            "Physical Type": column.physical_type,
            "Semantic Type": column.semantic_type,
            "Confidence": column.semantic_confidence,
            "Missing": column.missing_count,
            "Unique": column.unique_count,
            "Statistics": json.dumps(
                column.statistics,
                default=str
            )
        })

    st.subheader("Column Profiles")

    display_dataframe(
        pd.DataFrame(profile_rows),
        use_container_width=True,
        hide_index=True
    )


def display_schema_drift(services):
    st.subheader("Schema Drift Detection")

    st.write(
        "Compare the current dataset with an earlier CSV or Excel file "
        "to identify structural and semantic changes."
    )

    comparison_file = st.file_uploader(
        "Upload previous dataset for comparison",
        type=["csv", "xlsx", "xlsm"],
        key="schema_drift_upload"
    )

    if comparison_file is None:
        return

    signature = hashlib.sha256(
        comparison_file.getvalue()
    ).hexdigest()

    if st.session_state.schema_drift_signature != signature:
        st.session_state.schema_drift_signature = signature
        st.session_state.schema_drift_result = None

    try:
        previous_sheets = services["loader"].load(
            comparison_file,
            file_name=comparison_file.name
        )

        previous_names = list(previous_sheets)

        if not previous_names:
            st.warning("The comparison file contains no readable sheets.")
            return

        if len(previous_names) > 1:
            previous_sheet = st.selectbox(
                "Previous dataset sheet",
                previous_names,
                key="drift_previous_sheet"
            )
        else:
            previous_sheet = previous_names[0]

        if st.button(
            "Compare Schemas",
            key="compare_schema_drift"
        ):
            with st.spinner("Comparing dataset schemas..."):
                st.session_state.schema_drift_result = (
                    services["schema_drift"].compare(
                        previous_sheets[previous_sheet],
                        st.session_state.original_dataframe
                    )
                )

        result = st.session_state.schema_drift_result

        if result:
            st.metric(
                "Schema Drift",
                "Detected" if result.get("has_drift") else "No drift"
            )
            st.json(result)

    except Exception as error:
        st.error(f"Schema comparison failed: {error}")


def display_quality_report(report):
    st.subheader("Detected Data Quality Issues")

    issues = get_issues(report)

    if not issues:
        st.success("No data quality issues were detected.")
        return

    st.metric("Total Issues", len(issues))

    severity = report.severity_summary
    summary_columns = st.columns(4)

    for column, level in zip(
        summary_columns,
        ["low", "medium", "high", "critical"]
    ):
        column.metric(
            level.title(),
            severity.get(level, 0)
        )

    for index, issue in enumerate(issues, start=1):
        column_name = issue.get("column_name") or "Dataset"

        with st.expander(
            f"{index}. {issue['issue_type']} · "
            f"{column_name} · {issue['severity'].upper()}"
        ):
            st.write(issue.get("suggested_action", ""))

            st.write(
                f"Affected records: "
                f"{issue.get('affected_records', 0):,}"
            )

            st.write(
                f"Confidence: "
                f"{issue.get('confidence', 0):.0%}"
            )

            if issue.get("evidence"):
                st.json(issue["evidence"])


def display_advanced_intelligence(dataframe, services):
    st.subheader("Advanced Data Intelligence")

    if st.button(
        "Run Advanced Analysis",
        key="run_advanced_analysis"
    ):
        with st.spinner("Analyzing dataset..."):
            try:
                st.session_state.advanced_results = (
                    services["advanced"].analyze(dataframe)
                )
            except Exception as error:
                st.error(f"Advanced analysis failed: {error}")

    result = st.session_state.advanced_results

    if not result:
        return

    sections = [
        ("Text Analysis", "text_analysis"),
        ("Semantic Candidates", "semantic_candidates"),
        ("Anomalies", "anomalies"),
        ("Cross-Column Checks", "cross_column_checks")
    ]

    for title, key in sections:
        with st.expander(title):
            value = result.get(key, {})

            if not value:
                st.info("No results for this section.")

            elif isinstance(value, list):
                display_dataframe(
                    pd.DataFrame(value),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.json(value)


def display_privacy_panel(dataframe, services):
    st.subheader("Privacy and Sensitive Data")

    st.write(
        "Scan for possible sensitive columns. Choose which columns' "
        "statistics and inference evidence should be excluded "
        "from Gemini context."
    )

    if st.button(
        "Scan for Sensitive Data",
        key="scan_privacy"
    ):
        try:
            st.session_state.privacy_findings = (
                services["privacy"].detect(dataframe)
            )
            st.session_state.gemini_redacted_columns = []

        except Exception as error:
            st.error(f"Privacy scan failed: {error}")
            return

    findings = st.session_state.privacy_findings

    if not findings:
        st.info("Run a scan to inspect possible sensitive columns.")
        return

    display_dataframe(
        pd.DataFrame(findings),
        use_container_width=True,
        hide_index=True
    )

    candidates = list(dict.fromkeys(
        item["column"] for item in findings
    ))

    selected = st.multiselect(
        "Redact selected columns from Gemini context",
        candidates,
        default=st.session_state.gemini_redacted_columns,
        key="privacy_column_selector"
    )

    st.session_state.gemini_redacted_columns = selected

    st.caption(
        f"{len(selected)} column(s) selected for context redaction."
    )


def display_cleaning_plan(services):
    st.subheader("AI Cleaning Recommendations")

    if st.session_state.cleaning_plan is None:
        if st.button(
            "Generate Gemini Cleaning Plan",
            type="primary"
        ):
            try:
                from datadoctor.llm_engine import GeminiCleaningEngine

                with st.spinner(
                    "Gemini is analyzing the profiling results..."
                ):
                    llm = GeminiCleaningEngine()

                    plan = llm.generate_cleaning_plan(
                        st.session_state.original_profile,
                        st.session_state.quality_report,
                        redacted_columns=(
                            st.session_state.gemini_redacted_columns
                        )
                    )

                st.session_state.cleaning_plan = plan
                st.session_state.approved_issue_ids = []
                st.rerun()

            except Exception as error:
                st.error(f"Could not generate cleaning plan: {error}")

        return

    plan = st.session_state.cleaning_plan

    st.write(plan.dataset_summary)
    st.write(plan.overall_assessment)

    if plan.important_warnings:
        st.warning("Important warnings")

        for warning in plan.important_warnings:
            st.write(f"- {warning}")

    recommendations = get_recommendations(plan)

    if not recommendations:
        st.info("No cleaning recommendations were generated.")
        return

    cleaning_mode = st.radio(
        "Cleaning execution mode",
        [
            "Standard in-memory cleaning",
            "Chunk-by-chunk CSV cleaning"
        ],
        horizontal=True,
        key="cleaning_execution_mode"
    )

    st.session_state.chunked_mode = (
        cleaning_mode == "Chunk-by-chunk CSV cleaning"
    )

    if st.session_state.chunked_mode:
        st.number_input(
            "CSV chunk size",
            min_value=1000,
            max_value=500000,
            value=50000,
            step=1000,
            key="chunked_cleaning_size"
        )

        if not str(st.session_state.dataset_filename).lower().endswith(".csv"):
            st.warning(
                "Chunk-by-chunk execution is currently available only "
                "for CSV files. Choose standard cleaning for Excel."
            )

    labels = {}

    for recommendation in recommendations:
        issue_id = recommendation["issue_id"]

        label = (
            f"{recommendation.get('target_column') or 'Dataset'} · "
            f"{recommendation['operation']} · "
            f"{recommendation['explanation']}"
        )

        labels[label] = issue_id

        with st.expander(label):
            st.write(recommendation.get("rationale", ""))

            st.write(
                f"Expected benefit: "
                f"{recommendation.get('expected_benefit', '')}"
            )

            if recommendation.get("risks"):
                st.warning(
                    "Risks: " + "; ".join(recommendation["risks"])
                )

            if recommendation.get("parameters"):
                st.json(recommendation["parameters"])

    selected_labels = st.multiselect(
        "Approve recommendations individually",
        list(labels),
        default=[
            label
            for label, issue_id in labels.items()
            if issue_id in st.session_state.approved_issue_ids
        ],
        key="recommendation_approval_selector"
    )

    st.session_state.approved_issue_ids = [
        labels[label] for label in selected_labels
    ]

    approve_col, clear_col = st.columns(2)

    if approve_col.button(
        "Approve Complete Plan",
        use_container_width=True
    ):
        st.session_state.approved_issue_ids = list(
            dict.fromkeys(labels.values())
        )
        st.rerun()

    if clear_col.button(
        "Clear Approvals",
        use_container_width=True
    ):
        st.session_state.approved_issue_ids = []
        st.rerun()

    st.write(
        f"Approved recommendations: "
        f"{len(st.session_state.approved_issue_ids)}"
    )

    chunked_unavailable = (
        st.session_state.chunked_mode
        and not str(st.session_state.dataset_filename).lower().endswith(".csv")
    )

    if st.button(
        "Execute Approved Cleaning",
        type="primary",
        disabled=(
            not st.session_state.approved_issue_ids
            or chunked_unavailable
        )
    ):
        if st.session_state.chunked_mode:
            execute_chunked_cleaning(services)
        else:
            execute_cleaning(services)


def execute_cleaning(services):
    original = st.session_state.original_dataframe
    fingerprint = st.session_state.dataset_fingerprint

    try:
        with st.spinner(
            "Executing approved operations and validating the result..."
        ):
            cleaned, audit_log = services["executor"].execute(
                dataframe=original,
                profile=st.session_state.original_profile,
                report=st.session_state.quality_report,
                plan=st.session_state.cleaning_plan,
                approved_issue_ids=set(
                    st.session_state.approved_issue_ids
                ),
                expected_fingerprint=fingerprint
            )

            cleaned_profile = services["profiler"].profile_dataframe(
                cleaned,
                file_name=st.session_state.dataset_filename,
                sheet_name=st.session_state.selected_sheet
            )

            validation = services["validator"].validate(
                original=original,
                cleaned=cleaned,
                audit_log=audit_log,
                expected_fingerprint=fingerprint
            )

            after_fingerprint = DataIntegrity.fingerprint(cleaned)

            audit_metadata = services["audit"].create_record(
                filename=st.session_state.dataset_filename,
                sheet_name=st.session_state.selected_sheet,
                before_fingerprint=fingerprint,
                after_fingerprint=after_fingerprint,
                approved_issue_ids=st.session_state.approved_issue_ids,
                audit_log=audit_log,
                business_rules=st.session_state.configured_business_rules,
                validation=validation
            )

            services["history"].save_run(
                file_name=st.session_state.dataset_filename,
                sheet_name=st.session_state.selected_sheet,
                fingerprint_before=fingerprint,
                fingerprint_after=after_fingerprint,
                rows_before=len(original),
                rows_after=len(cleaned),
                approved_issue_ids=st.session_state.approved_issue_ids,
                audit_log=audit_log,
                metadata=audit_metadata
            )

            remove_temp_file(st.session_state.get("chunk_output_path"))

            st.session_state.chunk_output_path = None
            st.session_state.chunked_audit = []
            st.session_state.cleaned_dataframe = cleaned
            st.session_state.cleaned_profile = cleaned_profile
            st.session_state.audit_log = audit_log
            st.session_state.validation_results = validation
            st.session_state.cleaning_complete = True
            st.session_state.excel_report = None

        st.success(
            "Cleaning completed. Original data remains unchanged."
        )

    except Exception as error:
        st.error(f"Cleaning execution failed: {error}")


def execute_chunked_cleaning(services):
    source_path = st.session_state.source_temp_path

    if not source_path or not os.path.exists(source_path):
        st.error(
            "The original CSV file is no longer available. Upload it again."
        )
        return

    remove_temp_file(st.session_state.get("chunk_output_path"))

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix="_datadoctor_cleaned.csv",
        delete=False
    ) as temporary_file:
        output_path = temporary_file.name

    st.session_state.chunk_output_path = output_path

    fingerprint_before = file_sha256(source_path)
    chunksize = int(
        st.session_state.get("chunked_cleaning_size", 50000)
    )

    try:
        with st.spinner(
            "Cleaning the CSV incrementally, chunk by chunk..."
        ):
            result = services["chunked_executor"].execute_csv(
                input_path=source_path,
                output_path=output_path,
                plan=st.session_state.cleaning_plan,
                approved_issue_ids=set(
                    st.session_state.approved_issue_ids
                ),
                report=st.session_state.quality_report,
                chunksize=chunksize
            )

            if isinstance(result, tuple):
                audit_log = result[-1] if result else []
            elif isinstance(result, dict):
                audit_log = result.get("audit_log", [])
            elif isinstance(result, list):
                audit_log = result
            else:
                audit_log = []

            if not os.path.exists(output_path):
                raise RuntimeError(
                    "Chunked cleaning did not produce an output CSV file."
                )

            fingerprint_after = file_sha256(output_path)

            try:
                rows_after = count_csv_rows(output_path, chunksize)
            except pd.errors.EmptyDataError:
                rows_after = 0

            audit_metadata = services["audit"].create_record(
                filename=st.session_state.dataset_filename,
                sheet_name=st.session_state.selected_sheet,
                before_fingerprint=fingerprint_before,
                after_fingerprint=fingerprint_after,
                approved_issue_ids=st.session_state.approved_issue_ids,
                audit_log=audit_log,
                business_rules=st.session_state.configured_business_rules,
                validation={
                    "overall_status": "NOT_RUN",
                    "note": (
                        "Chunked output was generated incrementally. "
                        "Full in-memory validation was not run."
                    )
                }
            )

            services["history"].save_run(
                file_name=st.session_state.dataset_filename,
                sheet_name=st.session_state.selected_sheet,
                fingerprint_before=fingerprint_before,
                fingerprint_after=fingerprint_after,
                rows_before=len(st.session_state.original_dataframe),
                rows_after=rows_after,
                approved_issue_ids=st.session_state.approved_issue_ids,
                audit_log=audit_log,
                metadata=audit_metadata
            )

            st.session_state.chunked_audit = audit_log
            st.session_state.cleaned_dataframe = None
            st.session_state.cleaned_profile = None
            st.session_state.audit_log = audit_log
            st.session_state.validation_results = {
                "overall_status": "NOT_RUN",
                "total_checks": 0,
                "passed": 0,
                "warnings": 0,
                "failed": 0,
                "checks": [
                    {
                        "check": "Full dataset validation",
                        "status": "NOT_RUN",
                        "message": (
                            "Chunked cleaning completed. Full in-memory "
                            "validation is not performed in this mode."
                        )
                    }
                ]
            }
            st.session_state.cleaning_complete = True

        st.success(
            "Chunked cleaning completed. The original CSV remains unchanged."
        )
        st.rerun()

    except Exception as error:
        remove_temp_file(output_path)
        st.session_state.chunk_output_path = None
        st.error(f"Chunked cleaning failed: {error}")


def display_validation(validation):
    if not validation:
        st.info("Validation has not been performed.")
        return

    status = validation.get("overall_status", "UNKNOWN")

    if status == "PASSED":
        st.success("All validation checks passed.")
    elif status == "WARNING":
        st.warning("Validation completed with warnings.")
    elif status == "NOT_RUN":
        st.info("Full validation was not run in chunked mode.")
    else:
        st.error("One or more validation checks failed.")

    columns = st.columns(4)

    columns[0].metric(
        "Total Checks",
        validation.get("total_checks", 0)
    )
    columns[1].metric(
        "Passed",
        validation.get("passed", 0)
    )
    columns[2].metric(
        "Warnings",
        validation.get("warnings", 0)
    )
    columns[3].metric(
        "Failed",
        validation.get("failed", 0)
    )

    checks = validation.get("checks", [])

    if checks:
        display_dataframe(
            pd.DataFrame(checks),
            use_container_width=True,
            hide_index=True
        )


def profile_for_report(profile):
    return {
        "columns": [
            {
                "column_name": item.column_name,
                "semantic_type": item.semantic_type,
                "statistics": item.statistics
            }
            for item in profile.column_profiles.values()
        ]
    }


def display_cleaning_results(services):
    if not st.session_state.cleaning_complete:
        st.info(
            "Complete a cleaning run from Data Workspace "
            "to see results here."
        )
        return

    original = st.session_state.original_dataframe
    cleaned = st.session_state.cleaned_dataframe

    st.header("Cleaning Results")

    if cleaned is None and st.session_state.chunk_output_path:
        output_path = st.session_state.chunk_output_path

        if not os.path.exists(output_path):
            st.error(
                "The chunked output file is unavailable. Run cleaning again."
            )
            return

        rows_after = count_csv_rows(output_path)

        columns = st.columns(3)
        columns[0].metric("Rows Before", f"{len(original):,}")
        columns[1].metric("Rows After", f"{rows_after:,}")
        columns[2].metric(
            "Rows Removed",
            f"{max(0, len(original) - rows_after):,}"
        )

        st.info(
            "This result was produced with chunk-by-chunk CSV processing. "
            "The cleaned dataset is kept as a file rather than loaded fully "
            "into memory."
        )

        with open(output_path, "rb") as output_file:
            st.download_button(
                "Download Cleaned CSV",
                output_file.read(),
                file_name="datadoctor_cleaned.csv",
                mime="text/csv"
            )

        if st.session_state.chunked_audit:
            st.subheader("Chunked Cleaning Audit Log")

            display_dataframe(
                pd.DataFrame(st.session_state.chunked_audit),
                use_container_width=True,
                hide_index=True
            )

        display_validation(st.session_state.validation_results)
        return

    if cleaned is None:
        st.info("No cleaned dataset is available yet.")
        return

    columns = st.columns(4)

    columns[0].metric("Rows Before", f"{len(original):,}")
    columns[1].metric("Rows After", f"{len(cleaned):,}")
    columns[2].metric(
        "Rows Removed",
        f"{len(original) - len(cleaned):,}"
    )
    columns[3].metric(
        "Missing Cells Resolved",
        max(
            0,
            int(original.isna().sum().sum()) -
            int(cleaned.isna().sum().sum())
        )
    )

    tab_data, tab_validation, tab_audit, tab_export = st.tabs([
        "Cleaned Dataset",
        "Validation",
        "Audit Log",
        "Export Reports"
    ])

    with tab_data:
        display_dataframe(
            cleaned.head(200),
            use_container_width=True
        )

        st.download_button(
            "Download Cleaned CSV",
            cleaned.to_csv(index=False).encode("utf-8-sig"),
            file_name="datadoctor_cleaned.csv",
            mime="text/csv"
        )

    with tab_validation:
        display_validation(st.session_state.validation_results)

    with tab_audit:
        if st.session_state.audit_log:
            display_dataframe(
                pd.DataFrame(st.session_state.audit_log),
                use_container_width=True
            )
        else:
            st.info("No cleaning operations were recorded.")

    with tab_export:
        if st.button("Generate Excel Quality Report"):
            try:
                report_bytes = (
                    services["report_generator"].generate_excel(
                        original=original,
                        cleaned=cleaned,
                        original_profile=profile_for_report(
                            st.session_state.original_profile
                        ),
                        cleaned_profile=profile_for_report(
                            st.session_state.cleaned_profile
                        ),
                        quality_report=st.session_state.quality_report,
                        audit_log=st.session_state.audit_log,
                        validation=st.session_state.validation_results,
                        business_rule_results=(
                            st.session_state.business_rule_results
                        )
                    )
                )

                st.session_state.excel_report = report_bytes

            except Exception as error:
                st.error(f"Report generation failed: {error}")

        if st.session_state.excel_report:
            st.download_button(
                "Download Excel Quality Report",
                data=st.session_state.excel_report,
                file_name="datadoctor_quality_report.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


def display_business_rules(dataframe, services):
    st.header("Business Rule Validation")

    st.write(
        "Configure deterministic validation rules. These checks "
        "inspect the dataset and report violations; they do not modify data."
    )

    rules = services["rule_builder"].render(dataframe)

    if st.button(
        "Validate Business Rules",
        type="primary",
        disabled=not rules
    ):
        try:
            st.session_state.business_rule_results = (
                services["business_rules"].evaluate(
                    dataframe,
                    rules
                )
            )

        except Exception as error:
            st.session_state.business_rule_results = None
            st.error(f"Business rule evaluation failed: {error}")

    result = st.session_state.business_rule_results

    if not result:
        return

    st.divider()

    metrics = st.columns(4)

    metrics[0].metric("Total Rules", result.get("total_rules", 0))
    metrics[1].metric("Passed", result.get("rules_passed", 0))
    metrics[2].metric("Failed", result.get("rules_failed", 0))
    metrics[3].metric("Violations", result.get("total_violations", 0))

    for index, rule_result in enumerate(
        result.get("results", []),
        start=1
    ):
        title = rule_result.get("rule_name", f"Rule {index}")

        with st.expander(
            f"{title} · {rule_result.get('status', 'UNKNOWN')}"
        ):
            st.write(rule_result.get("message", ""))

            st.write(
                f"Violation count: "
                f"{rule_result.get('violation_count', 0)}"
            )

            samples = rule_result.get("sample_values", [])

            if samples:
                display_dataframe(
                    pd.DataFrame(samples),
                    use_container_width=True,
                    hide_index=True
                )

    csv_data = pd.DataFrame(
        result.get("results", [])
    ).to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Download Business Rule Results",
        csv_data,
        file_name="datadoctor_business_rule_results.csv",
        mime="text/csv"
    )


def display_history(services):
    st.header("Cleaning History")

    try:
        history = services["history"].get_recent_runs(limit=50)

        if not history:
            st.info("No previous cleaning runs found.")
            return

        display = pd.DataFrame(history)

        display_dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        if "id" in display.columns:
            run_id = st.selectbox(
                "Inspect run",
                display["id"].tolist()
            )

            selected = services["history"].get_run(int(run_id))

            if selected:
                with st.expander("Audit metadata"):
                    try:
                        st.json(
                            json.loads(
                                selected.get("metadata", "{}")
                            )
                        )
                    except (TypeError, json.JSONDecodeError):
                        st.write(selected.get("metadata", "{}"))

    except Exception as error:
        st.error(f"Could not retrieve cleaning history: {error}")


def main():
    initialize_state()
    services = get_services()

    st.title("🩺 DataDoctor AI")
    st.caption(
        "Intelligent Data Profiling, Cleaning and Quality Assurance"
    )

    with st.sidebar:
        st.title("Navigation")

        page = st.radio(
            "Choose a section",
            [
                "Data Workspace",
                "Cleaning Results",
                "Cleaning History"
            ]
        )

        st.divider()

        if st.button(
            "Reset Current Dataset",
            use_container_width=True
        ):
            reset_analysis()

        st.caption("DataDoctor AI · Phase 8")

    if page == "Data Workspace":
        st.header("Dataset Workspace")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel dataset",
            type=["csv", "xlsx", "xlsm"]
        )

        if uploaded_file is not None:
            content_hash = hashlib.sha256(
                uploaded_file.getvalue()
            ).hexdigest()

            signature = (
                uploaded_file.name,
                uploaded_file.size,
                content_hash
            )

            if st.session_state.last_uploaded_signature != signature:
                remove_temp_file(
                    st.session_state.get("source_temp_path")
                )
                remove_temp_file(
                    st.session_state.get("chunk_output_path")
                )

                for key in [
                    "original_dataframe",
                    "cleaned_dataframe",
                    "original_profile",
                    "cleaned_profile",
                    "quality_report",
                    "cleaning_plan",
                    "approved_issue_ids",
                    "audit_log",
                    "validation_results",
                    "business_rule_results",
                    "dataset_fingerprint",
                    "dataset_filename",
                    "selected_sheet",
                    "analysis_complete",
                    "cleaning_complete",
                    "configured_business_rules",
                    "privacy_findings",
                    "gemini_redacted_columns",
                    "advanced_results",
                    "excel_report",
                    "source_temp_path",
                    "chunk_output_path",
                    "chunked_audit",
                    "chunked_mode",
                    "schema_drift_result",
                    "schema_drift_signature"
                ]:
                    st.session_state.pop(key, None)

                st.session_state.last_uploaded_signature = signature
                initialize_state()
                st.session_state.last_uploaded_signature = signature
                save_uploaded_source(uploaded_file)

            if (
                not st.session_state.get("source_temp_path")
                or not os.path.exists(st.session_state.source_temp_path)
            ):
                save_uploaded_source(uploaded_file)

            try:
                datasets = load_dataset(uploaded_file, services)
                sheet_names = list(datasets)

                if not sheet_names:
                    st.error("No readable data was found in this file.")
                    return

                if len(sheet_names) > 1:
                    sheet_name = st.selectbox(
                        "Select Excel sheet",
                        sheet_names
                    )
                else:
                    sheet_name = sheet_names[0]

                dataframe = datasets[sheet_name]

                st.success(
                    f"Loaded {uploaded_file.name} · {sheet_name}"
                )

                st.subheader("Dataset Preview")

                display_dataframe(
                    dataframe.head(20),
                    use_container_width=True
                )

                if st.button("Analyze Dataset", type="primary"):
                    with st.spinner(
                        "Profiling dataset and detecting quality issues..."
                    ):
                        analyze_dataset(
                            dataframe,
                            uploaded_file.name,
                            sheet_name,
                            services
                        )

                    st.success("Dataset analysis completed.")
                    st.rerun()

            except Exception as error:
                st.error(f"Dataset loading failed: {error}")

        if st.session_state.analysis_complete:
            dataframe = st.session_state.original_dataframe

            st.divider()
            display_dataset_overview(
                dataframe,
                st.session_state.original_profile,
                services
            )

            st.divider()
            display_schema_drift(services)

            st.divider()
            display_quality_report(st.session_state.quality_report)

            st.divider()
            display_advanced_intelligence(dataframe, services)

            st.divider()
            display_privacy_panel(dataframe, services)

            st.divider()
            display_cleaning_plan(services)

            st.divider()
            display_business_rules(dataframe, services)

    elif page == "Cleaning Results":
        display_cleaning_results(services)

    else:
        display_history(services)


if __name__ == "__main__":
    main()