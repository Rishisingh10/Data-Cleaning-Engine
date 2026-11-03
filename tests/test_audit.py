from datadoctor.audit import AuditManager


def test_canonical_hash_is_order_independent_for_dicts():
    manager = AuditManager()
    assert manager.canonical_hash({"a": 1, "b": 2}) == manager.canonical_hash({"b": 2, "a": 1})


def test_audit_record_contains_reproducibility_metadata():
    record = AuditManager().create_record(
        "data.csv", "CSV", "before", "after", ["issue_1"], [],
        business_rules=[{"type": "required", "column": "email"}],
        validation={"overall_status": "PASSED"}
    )
    assert record["before_fingerprint"] == "before"
    assert record["after_fingerprint"] == "after"
    assert record["business_rules_hash"]
    assert record["validation_hash"]