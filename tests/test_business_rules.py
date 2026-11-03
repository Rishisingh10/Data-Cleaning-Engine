
import pandas as pd
import pytest

from datadoctor.business_rules import (
    BusinessRuleEngine,
    BusinessRuleError
)


def test_age_range_rule():
    dataframe = pd.DataFrame({
        "Age": [20, 35, 250, -5]
    })

    rules = [{
        "type": "range",
        "column": "Age",
        "min": 0,
        "max": 120
    }]

    result = BusinessRuleEngine().evaluate(dataframe, rules)

    assert result["total_violations"] == 2


def test_order_delivery_date_rule():
    dataframe = pd.DataFrame({
        "Order Date": ["2026-01-10", "2026-01-15"],
        "Delivery Date": ["2026-01-12", "2026-01-13"]
    })

    rules = [{
        "type": "comparison",
        "left_column": "Order Date",
        "operator": "<=",
        "right_column": "Delivery Date",
        "comparison_type": "date"
    }]

    result = BusinessRuleEngine().evaluate(dataframe, rules)

    assert result["total_violations"] == 1


def test_country_currency_mapping():
    dataframe = pd.DataFrame({
        "Country": ["India", "USA", "India"],
        "Currency": ["INR", "USD", "USD"]
    })

    rules = [{
        "type": "mapping",
        "key_column": "Country",
        "value_column": "Currency",
        "allowed_mappings": {
            "India": ["INR"],
            "USA": ["USD"]
        }
    }]

    result = BusinessRuleEngine().evaluate(dataframe, rules)

    assert result["total_violations"] == 1


def test_allowed_values_rule():
    dataframe = pd.DataFrame({
        "Status": ["Pending", "Shipped", "Unknown"]
    })

    rules = [{
        "type": "allowed_values",
        "column": "Status",
        "values": ["Pending", "Shipped", "Delivered"]
    }]

    result = BusinessRuleEngine().evaluate(dataframe, rules)

    assert result["total_violations"] == 1


def test_required_rule():
    dataframe = pd.DataFrame({
        "Customer ID": ["C101", "", None, "C104"]
    })

    rules = [{
        "type": "required",
        "column": "Customer ID"
    }]

    result = BusinessRuleEngine().evaluate(dataframe, rules)

    assert result["total_violations"] == 2


def test_unique_rule():
    dataframe = pd.DataFrame({
        "Transaction ID": ["T1", "T2", "T1", "T3"]
    })

    rules = [{
        "type": "unique",
        "column": "Transaction ID"
    }]

    result = BusinessRuleEngine().evaluate(dataframe, rules)

    assert result["total_violations"] == 2


def test_missing_column_raises_error():
    dataframe = pd.DataFrame({
        "Age": [20, 30]
    })

    rules = [{
        "type": "range",
        "column": "Salary",
        "min": 0,
        "max": 100000
    }]

    with pytest.raises(BusinessRuleError):
        BusinessRuleEngine().evaluate(dataframe, rules)


def test_invalid_rule_type_raises_error():
    dataframe = pd.DataFrame({
        "Age": [20, 30]
    })

    rules = [{
        "type": "unknown",
        "column": "Age"
    }]

    with pytest.raises(BusinessRuleError):
        BusinessRuleEngine().evaluate(dataframe, rules)


def test_business_rules_do_not_modify_dataframe():
    dataframe = pd.DataFrame({
        "Age": [20, 250]
    })

    original = dataframe.copy(deep=True)

    rules = [{
        "type": "range",
        "column": "Age",
        "min": 0,
        "max": 120
    }]

    BusinessRuleEngine().evaluate(dataframe, rules)

    pd.testing.assert_frame_equal(dataframe, original)
