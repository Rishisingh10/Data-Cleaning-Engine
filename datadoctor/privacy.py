import re
from collections import defaultdict

import pandas as pd


class PrivacyManager:
    PATTERNS = {
        "email": re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.I),
        "phone": re.compile(r"^\+?[\d\s().-]{7,}$"),
        "aadhaar_candidate": re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$"),
        "credit_card_candidate": re.compile(r"^(?:\d[ -]?){13,19}$"),
        "ip_address": re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    }

    NAME_HINTS = {
        "email": ("email", "e-mail", "mail"),
        "phone": ("phone", "mobile", "contact", "telephone"),
        "aadhaar_candidate": ("aadhaar", "aadhar", "uid"),
        "credit_card_candidate": ("card", "credit"),
        "ip_address": ("ip_address", "ip")
    }

    def detect(self, dataframe, sample_size=500):
        findings = []
        for column in dataframe.columns:
            series = dataframe[column]
            name = str(column).casefold()
            sample = series.dropna().astype(str).head(sample_size).str.strip()
            if sample.empty:
                continue
            for kind, pattern in self.PATTERNS.items():
                ratio = float(sample.map(lambda value: bool(pattern.fullmatch(value))).mean())
                hinted = any(hint in name for hint in self.NAME_HINTS[kind])
                if ratio >= 0.5 or (hinted and ratio > 0):
                    findings.append({
                        "column": str(column),
                        "sensitive_type": kind,
                        "match_ratio": round(ratio, 4),
                        "sample_size": len(sample),
                        "name_hint": hinted,
                        "requires_review": True
                    })
        return findings

    def redact_context(self, context, sensitive_columns, replacement="[REDACTED]"):
        sensitive = {str(column) for column in sensitive_columns}

        def redact(value):
            if isinstance(value, dict):
                if str(value.get("name", "")) in sensitive:
                    return {
                        key: replacement if key in {
                            "statistics", "inference_evidence", "samples",
                            "top_values", "most_common_values"
                        } else redact(item)
                        for key, item in value.items()
                    }
                return {key: redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        return redact(context)

    def redact_dataframe(self, dataframe, columns, replacement="[REDACTED]"):
        result = dataframe.copy(deep=True)
        valid = [column for column in columns if column in result.columns]
        for column in valid:
            result[column] = result[column].where(result[column].isna(), replacement)
        return result