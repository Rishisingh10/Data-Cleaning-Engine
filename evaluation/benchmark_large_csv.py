
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datadoctor.chunked_executor import ChunkedCleaningExecutor


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "test_datasets" / "nyc_311_large.csv"
OUTPUT_DIR = BASE_DIR / "benchmark_outputs"
OUTPUT_FILE = OUTPUT_DIR / "nyc_311_cleaned.csv"
RESULT_FILE = OUTPUT_DIR / "benchmark_results.json"

CHUNK_SIZE = 50000
ISSUE_ID = "BENCHMARK-DUPLICATES"


class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.peak_memory = 0
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            try:
                memory = self.process.memory_info().rss
                self.peak_memory = max(self.peak_memory, memory)
            except psutil.Error:
                pass

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.peak_memory = self.process.memory_info().rss
        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join()

        self.peak_memory = max(
            self.peak_memory,
            self.process.memory_info().rss,
        )

        return round(self.peak_memory / (1024 * 1024), 2)


def run_benchmark():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recommendation = SimpleNamespace(
        issue_id=ISSUE_ID,
        operation="remove_exact_duplicates",
        target_column=None,
        parameters={},
    )

    issue = SimpleNamespace(
        issue_id=ISSUE_ID,
        issue_type="duplicate_rows",
    )

    plan = SimpleNamespace(
        recommendations=[recommendation],
    )

    report = SimpleNamespace(
        issues=[issue],
    )

    executor = ChunkedCleaningExecutor()
    monitor = MemoryMonitor()

    input_size_mb = round(
        INPUT_FILE.stat().st_size / (1024 * 1024),
        2,
    )

    print("DATADOCTOR LARGE CSV BENCHMARK")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Input size: {input_size_mb} MB")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print("Operation: remove_exact_duplicates")
    print("Starting benchmark...\n")

    monitor.start()

    try:
        result = executor.execute_csv(
            input_path=INPUT_FILE,
            output_path=OUTPUT_FILE,
            plan=plan,
            approved_issue_ids=[ISSUE_ID],
            report=report,
            chunksize=CHUNK_SIZE,
        )
    finally:
        peak_memory_mb = monitor.stop()

    output_size_mb = round(
        OUTPUT_FILE.stat().st_size / (1024 * 1024),
        2,
    )

    benchmark_results = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb,
        "chunk_size": CHUNK_SIZE,
        "rows_read": result["rows_read"],
        "rows_written": result["rows_written"],
        "duplicates_removed": result["duplicates_removed"],
        "execution_time_seconds": result["execution_time_seconds"],
        "peak_process_memory_mb": peak_memory_mb,
        "rows_per_second": result["rows_per_second"],
        "input_sha256": result["input_sha256"],
        "output_sha256": result["output_sha256"],
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            benchmark_results,
            file,
            indent=4,
        )

    print("BENCHMARK RESULTS")
    print("=" * 60)

    for key, value in benchmark_results.items():
        print(f"{key}: {value}")

    print("\nResults saved to:", RESULT_FILE)


if __name__ == "__main__":
    run_benchmark()
