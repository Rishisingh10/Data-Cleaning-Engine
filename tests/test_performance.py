import pandas as pd

from datadoctor.performance import DatasetPerformance


def test_assess_reports_dimensions_and_memory():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = DatasetPerformance().assess(frame)
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["memory_bytes"] > 0
    assert result["tier"] == "small"


def test_sample_is_deterministic_and_bounded():
    frame = pd.DataFrame({"value": range(100)})
    sample = DatasetPerformance().sample(frame, sample_size=10)
    assert len(sample) == 10
    assert sample.equals(DatasetPerformance().sample(frame, sample_size=10))