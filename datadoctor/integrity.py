
import hashlib
import json

import pandas as pd


class DataIntegrityError(Exception):
    pass


class DataIntegrity:

    @staticmethod
    def fingerprint(dataframe: pd.DataFrame) -> str:
        metadata = {
            "columns": [str(column) for column in dataframe.columns],
            "dtypes": [str(dtype) for dtype in dataframe.dtypes],
            "shape": dataframe.shape
        }

        try:
            row_hashes = pd.util.hash_pandas_object(
                dataframe,
                index=True,
                categorize=True
            )

            digest = hashlib.sha256()
            digest.update(json.dumps(metadata, sort_keys=True).encode())
            digest.update(row_hashes.values.tobytes())

            return digest.hexdigest()

        except Exception as error:
            raise DataIntegrityError(
                f"Could not fingerprint dataset: {error}"
            ) from error

    @staticmethod
    def validate_structure(
        original: pd.DataFrame,
        cleaned: pd.DataFrame
    ) -> None:
        if list(original.columns) != list(cleaned.columns):
            raise DataIntegrityError(
                "Cleaning changed the dataset columns."
            )

        if list(original.dtypes.astype(str)) != list(
            cleaned.dtypes.astype(str)
        ):
            raise DataIntegrityError(
                "Cleaning unexpectedly changed column data types."
            )

    @staticmethod
    def validate_original(
        dataframe: pd.DataFrame,
        expected_fingerprint: str
    ) -> None:
        actual_fingerprint = DataIntegrity.fingerprint(dataframe)

        if actual_fingerprint != expected_fingerprint:
            raise DataIntegrityError(
                "Original dataset was modified unexpectedly."
            )

    @staticmethod
    def rollback(original: pd.DataFrame) -> pd.DataFrame:
        return original.copy(deep=True)
