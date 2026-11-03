import pandas as pd

from datadoctor.privacy import PrivacyManager


def test_detects_email_column():
    frame = pd.DataFrame({"email": ["a@example.com", "b@example.com", "bad"]})
    findings = PrivacyManager().detect(frame)
    assert findings
    assert findings[0]["sensitive_type"] == "email"


def test_redacts_selected_context_fields():
    context = {"columns": [{"name": "email", "statistics": {"top": "a@example.com"}}]}
    result = PrivacyManager().redact_context(context, ["email"])
    assert result["columns"][0]["statistics"] == "[REDACTED]"


def test_dataframe_redaction_preserves_missing_values():
    frame = pd.DataFrame({"email": ["a@example.com", None]})
    result = PrivacyManager().redact_dataframe(frame, ["email"])
    assert result["email"].iloc[0] == "[REDACTED]"
    assert pd.isna(result["email"].iloc[1])
    assert frame["email"].iloc[0] == "a@example.com"