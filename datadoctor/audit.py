import hashlib
import json
from datetime import datetime, timezone


class AuditManager:
    @staticmethod
    def canonical_hash(value):
        payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def create_record(self, filename, sheet_name, before_fingerprint,
                      after_fingerprint, approved_issue_ids, audit_log,
                      business_rules=None, validation=None):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "sheet_name": sheet_name,
            "before_fingerprint": before_fingerprint,
            "after_fingerprint": after_fingerprint,
            "approved_issue_ids": list(approved_issue_ids or []),
            "audit_log": audit_log or [],
            "business_rules_hash": self.canonical_hash(business_rules or []),
            "validation_hash": self.canonical_hash(validation or {}),
        }