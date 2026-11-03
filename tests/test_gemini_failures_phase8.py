import pandas as pd
import pytest

from datadoctor.llm_engine import (
    GeminiCleaningEngine,
    GeminiEngineError
)

from datadoctor.profiler import DataProfiler
from datadoctor.detectors import DataQualityDetector


class FailingModels:

    def generate_content(self, **kwargs):
        raise RuntimeError("400 invalid request")


class FailingClient:
    models = FailingModels()


class FakeTypes:

    @staticmethod
    def GenerateContentConfig(**kwargs):
        return kwargs


def test_gemini_non_retryable_failure_is_wrapped():
    frame = pd.DataFrame({
        "name": ["A", None]
    })

    profile = DataProfiler().profile_dataframe(frame)

    report = DataQualityDetector().detect(
        frame,
        profile
    )

    engine = GeminiCleaningEngine.__new__(
        GeminiCleaningEngine
    )

    engine.client = FailingClient()
    engine.types = FakeTypes()
    engine.model = "test-model"

    with pytest.raises(
        GeminiEngineError,
        match="Gemini API request failed"
    ):
        engine.generate_cleaning_plan(
            profile,
            report
        )


def test_gemini_rejects_profile_report_mismatch():
    frame = pd.DataFrame({
        "name": ["A", None]
    })

    profile = DataProfiler().profile_dataframe(frame)

    report = DataQualityDetector().detect(
        frame,
        profile
    )

    other_profile = DataProfiler().profile_dataframe(
        pd.DataFrame({"age": [1, 2]})
    )

    engine = GeminiCleaningEngine.__new__(
        GeminiCleaningEngine
    )

    engine.client = FailingClient()
    engine.types = FakeTypes()
    engine.model = "test-model"

    with pytest.raises(
        GeminiEngineError,
        match="do not match"
    ):
        engine.generate_cleaning_plan(
            other_profile,
            report
        )