import pandas as pd


class DatasetPerformance:
    DEFAULT_SAMPLE_SIZE = 10000

    def assess(self, dataframe):
        memory_bytes = int(dataframe.memory_usage(index=True, deep=True).sum())
        rows, columns = dataframe.shape

        if memory_bytes < 50 * 1024 * 1024:
            tier = "small"
        elif memory_bytes < 500 * 1024 * 1024:
            tier = "medium"
        else:
            tier = "large"

        return {
            "rows": int(rows),
            "columns": int(columns),
            "memory_bytes": memory_bytes,
            "memory_mb": round(memory_bytes / (1024 ** 2), 2),
            "tier": tier,
            "recommended_sample_size": min(rows, self.DEFAULT_SAMPLE_SIZE)
        }

    def sample(self, dataframe, sample_size=None, random_state=42):
        size = sample_size or self.DEFAULT_SAMPLE_SIZE

        if len(dataframe) <= size:
            return dataframe

        return dataframe.sample(n=size, random_state=random_state)