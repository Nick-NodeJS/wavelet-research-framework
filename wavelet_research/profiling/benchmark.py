"""Benchmarking framework for CPU, memory, latency, and throughput.

Provides deterministic benchmarking tools for profiling
engine performance and detecting regressions.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single benchmark run.

    Parameters
    ----------
    name : str
        Benchmark name.
    mean_time_ms : float
        Mean execution time in milliseconds.
    std_time_ms : float
        Standard deviation of execution time.
    min_time_ms : float
        Minimum execution time.
    max_time_ms : float
        Maximum execution time.
    iterations : int
        Number of iterations.
    throughput_per_sec : float
        Operations per second.
    peak_memory_kb : float
        Peak memory usage in KB.
    """

    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    throughput_per_sec: float
    peak_memory_kb: float


class LatencyBenchmark:
    """Measures execution latency of a callable.

    Parameters
    ----------
    name : str
        Benchmark name.
    func : Callable
        Function to benchmark.
    iterations : int
        Number of iterations to run.
    warmup : int
        Number of warmup iterations.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[], None],
        iterations: int = 100,
        warmup: int = 10,
    ) -> None:
        self._name = name
        self._func = func
        self._iterations = iterations
        self._warmup = warmup

    def run(self) -> BenchmarkResult:
        """Execute the benchmark.

        Returns
        -------
        BenchmarkResult
            Benchmark results.
        """
        # Warmup
        for _ in range(self._warmup):
            self._func()

        times: list[float] = []
        for _ in range(self._iterations):
            start = time.perf_counter_ns()
            self._func()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        arr = np.asarray(times, dtype=np.float64)
        mean = float(np.mean(arr))
        throughput = 1000.0 / mean if mean > 0 else 0.0

        return BenchmarkResult(
            name=self._name,
            mean_time_ms=mean,
            std_time_ms=float(np.std(arr)),
            min_time_ms=float(np.min(arr)),
            max_time_ms=float(np.max(arr)),
            iterations=self._iterations,
            throughput_per_sec=throughput,
            peak_memory_kb=0.0,
        )


class MemoryBenchmark:
    """Measures peak memory usage of a callable.

    Parameters
    ----------
    name : str
        Benchmark name.
    func : Callable
        Function to benchmark.
    """

    def __init__(self, name: str, func: Callable[[], None]) -> None:
        self._name = name
        self._func = func

    def run(self) -> BenchmarkResult:
        """Execute the memory benchmark.

        Returns
        -------
        BenchmarkResult
            Result with peak memory.
        """
        tracemalloc.start()
        start = time.perf_counter_ns()
        self._func()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_kb = peak / 1024

        return BenchmarkResult(
            name=self._name,
            mean_time_ms=elapsed_ms,
            std_time_ms=0.0,
            min_time_ms=elapsed_ms,
            max_time_ms=elapsed_ms,
            iterations=1,
            throughput_per_sec=1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
            peak_memory_kb=peak_kb,
        )


class ThroughputBenchmark:
    """Measures throughput (items/sec) for a processing function.

    Parameters
    ----------
    name : str
        Benchmark name.
    func : Callable
        Function that processes items (should be per-item callable).
    n_items : int
        Number of items to process.
    """

    def __init__(
        self, name: str, func: Callable[[], None], n_items: int
    ) -> None:
        self._name = name
        self._func = func
        self._n_items = n_items

    def run(self) -> BenchmarkResult:
        """Execute throughput benchmark.

        Returns
        -------
        BenchmarkResult
            Result with throughput.
        """
        start = time.perf_counter_ns()
        self._func()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        elapsed_sec = elapsed_ms / 1000.0
        throughput = self._n_items / elapsed_sec if elapsed_sec > 0 else 0.0

        return BenchmarkResult(
            name=self._name,
            mean_time_ms=elapsed_ms,
            std_time_ms=0.0,
            min_time_ms=elapsed_ms,
            max_time_ms=elapsed_ms,
            iterations=1,
            throughput_per_sec=throughput,
            peak_memory_kb=0.0,
        )


class BenchmarkSuite:
    """Suite of benchmarks with regression detection.

    Parameters
    ----------
    name : str
        Suite name.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._results: list[BenchmarkResult] = []
        self._baselines: dict[str, float] = {}

    @property
    def name(self) -> str:
        """Suite name."""
        return self._name

    @property
    def results(self) -> list[BenchmarkResult]:
        """All benchmark results."""
        return list(self._results)

    def set_baseline(self, benchmark_name: str, time_ms: float) -> None:
        """Set a performance baseline for regression detection.

        Parameters
        ----------
        benchmark_name : str
            Name of the benchmark.
        time_ms : float
            Baseline time in milliseconds.
        """
        self._baselines[benchmark_name] = time_ms

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result.

        Parameters
        ----------
        result : BenchmarkResult
            Benchmark result to add.
        """
        self._results.append(result)

    def check_regressions(self, tolerance: float = 0.2) -> list[str]:
        """Check for performance regressions against baselines.

        Parameters
        ----------
        tolerance : float
            Allowed deviation fraction (0.2 = 20% slower allowed).

        Returns
        -------
        list[str]
            List of regression messages. Empty = no regressions.
        """
        regressions: list[str] = []
        for result in self._results:
            if result.name in self._baselines:
                baseline = self._baselines[result.name]
                if result.mean_time_ms > baseline * (1 + tolerance):
                    regressions.append(
                        f"{result.name}: {result.mean_time_ms:.2f}ms "
                        f"exceeds baseline {baseline:.2f}ms "
                        f"(+{(result.mean_time_ms / baseline - 1) * 100:.1f}%)"
                    )
        return regressions
