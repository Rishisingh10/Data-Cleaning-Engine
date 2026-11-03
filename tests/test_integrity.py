
import pandas as pd
import pytest

from datadoctor.integrity import (
    DataIntegrity,
    DataIntegrityError
)


def test_identical_dataframes_have_same_fingerprint():
    first = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    second = first.copy(deep=True)

    assert DataIntegrity.fingerprint(first) == DataIntegrity.fingerprint(second)


def test_modified_dataframe_has_different_fingerprint():
    first = pd.DataFrame({"A": [1, 2]})
    second = first.copy(deep=True)
    second.loc[0, "A"] = 100

    assert DataIntegrity.fingerprint(first) != DataIntegrity.fingerprint(second)


def test_index_changes_fingerprint():
    first = pd.DataFrame({"A": [1, 2]})
    second = first.copy(deep=True)
    second.index = [10, 11]

    assert DataIntegrity.fingerprint(first) != DataIntegrity.fingerprint(second)


def test_original_dataset_validation_passes():
    dataframe = pd.DataFrame({"A": [1, 2]})
    fingerprint = DataIntegrity.fingerprint(dataframe)

    DataIntegrity.validate_original(dataframe, fingerprint)


def test_original_dataset_validation_rejects_changes():
    dataframe = pd.DataFrame({"A": [1, 2]})
    fingerprint = DataIntegrity.fingerprint(dataframe)
    dataframe.loc[0, "A"] = 99

    with pytest.raises(DataIntegrityError):
        DataIntegrity.validate_original(dataframe, fingerprint)


def test_structure_validation_rejects_changed_columns():
    original = pd.DataFrame({"A": [1], "B": [2]})
    cleaned = pd.DataFrame({"A": [1], "C": [2]})

    with pytest.raises(DataIntegrityError):
        DataIntegrity.validate_structure(original, cleaned)


def test_rollback_restores_original_values():
    original = pd.DataFrame({"A": [1, 2]})
    cleaned = pd.DataFrame({"A": [10, 20]})

    restored = DataIntegrity.rollback(original)

    pd.testing.assert_frame_equal(restored, original)
    assert not restored.equals(cleaned)


def test_rollback_does_not_modify_original():
    original = pd.DataFrame({"A": [1, 2]})
    snapshot = original.copy(deep=True)

    restored = DataIntegrity.rollback(original)
    restored.loc[0, "A"] = 99

    pd.testing.assert_frame_equal(original, snapshot)
