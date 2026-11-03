
import pandas as pd

from datadoctor.advanced_intelligence import AdvancedDataIntelligence


def test_text_analysis():
    dataframe = pd.DataFrame({
        "Description": [
            "Hello world",
            "Data   cleaning",
            " Short text "
        ]
    })

    result = AdvancedDataIntelligence().analyze(dataframe)

    details = result["text_analysis"]["Description"]

    assert details["non_null_count"] == 3
    assert details["repeated_whitespace"] == 1
    assert details["leading_trailing_spaces"] == 1


def test_empty_string_detection():
    dataframe = pd.DataFrame({
        "Description": ["Hello", "", "   ", None]
    })

    result = AdvancedDataIntelligence().analyze(dataframe)

    assert result["text_analysis"]["Description"]["empty_string_count"] == 2


def test_email_candidate_detection():
    dataframe = pd.DataFrame({
        "Email": [
            "a@example.com",
            "b@example.com",
            "invalid-email"
        ]
    })

    result = AdvancedDataIntelligence().analyze(dataframe)

    assert (
        result["semantic_candidates"]["Email"]["candidate_type"]
        == "email"
    )


def test_numeric_text_detection():
    dataframe = pd.DataFrame({
        "Score": ["10", "20", "30", "40"]
    })

    result = AdvancedDataIntelligence().analyze(dataframe)

    assert (
        result["semantic_candidates"]["Score"]["candidate_type"]
        == "numerical_text"
    )


def test_outlier_detection():
    dataframe = pd.DataFrame({
        "Amount": [10, 11, 12, 13, 14, 1000]
    })

    result = AdvancedDataIntelligence().analyze(dataframe)

    assert any(
        item["anomaly_type"] == "statistical_outlier"
        for item in result["anomalies"]
    )


def test_identical_column_detection():
    dataframe = pd.DataFrame({
        "Age": [20, 30, 40],
        "Age_Copy": [20, 30, 40]
    })

    result = AdvancedDataIntelligence().analyze(dataframe)

    assert any(
        item["check_type"] == "identical_columns"
        for item in result["cross_column_checks"]
    )


def test_high_correlation_detection():
    dataframe = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": [2, 4, 6, 8, 10, 12]
    })

    result = AdvancedDataIntelligence().analyze(dataframe)

    assert any(
        item["check_type"] == "high_correlation"
        for item in result["cross_column_checks"]
    )


def test_original_dataframe_unchanged():
    dataframe = pd.DataFrame({
        "Description": [" Hello ", "World"],
        "Amount": [10, 20]
    })

    original = dataframe.copy(deep=True)

    AdvancedDataIntelligence().analyze(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)
