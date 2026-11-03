
from pathlib import Path
from typing import BinaryIO

import pandas as pd


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded."""


class DatasetLoader:
    """Loads supported files into Pandas DataFrames."""

    SUPPORTED_FORMATS = {".csv", ".xlsx", ".xlsm"}

    def load(
        self,
        file: str | Path | BinaryIO,
        file_name: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Load a dataset.

        Returns a dictionary mapping sheet names to DataFrames.
        CSV files are represented as a single 'Sheet1'.
        """

        filename, source = self._resolve_source(file, file_name)
        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            raise DatasetLoadError(
                f"Unsupported file format: {extension or 'unknown'}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        try:
            if extension == ".csv":
                return {"Sheet1": self._load_csv(source)}

            return self._load_excel(source)

        except DatasetLoadError:
            raise

        except Exception as error:
            raise DatasetLoadError(
                f"Could not load '{filename}': {error}"
            ) from error

    @staticmethod
    def _resolve_source(
        file: str | Path | BinaryIO,
        file_name: str | None
    ):
        """Resolve the filename and validate the input source."""

        if isinstance(file, (str, Path)):
            path = Path(file)

            if not path.is_file():
                raise DatasetLoadError(
                    f"File does not exist: {path}"
                )

            return path.name, path

        if not file_name:
            raise DatasetLoadError(
                "A filename is required for uploaded files."
            )

        if not hasattr(file, "read"):
            raise DatasetLoadError(
                "Input must be a file path or a readable binary file."
            )

        return Path(file_name).name, file

    @staticmethod
    def _reset_stream(source):
        """Move uploaded file streams back to their beginning."""

        if hasattr(source, "seek"):
            source.seek(0)

    def _load_csv(self, source) -> pd.DataFrame:
        """Read CSV while handling common text encodings."""

        self._reset_stream(source)

        try:
            return pd.read_csv(source, encoding="utf-8-sig")
        except UnicodeDecodeError:
            self._reset_stream(source)

            try:
                return pd.read_csv(source, encoding="cp1252")
            except UnicodeDecodeError as error:
                raise DatasetLoadError(
                    "CSV encoding is not supported. "
                    "Please save the file as UTF-8."
                ) from error

    def _load_excel(
        self,
        source
    ) -> dict[str, pd.DataFrame]:
        """Read every worksheet in an Excel workbook."""

        self._reset_stream(source)

        return pd.read_excel(
            source,
            sheet_name=None,
            engine="openpyxl"
        )
