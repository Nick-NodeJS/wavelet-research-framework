"""Tests for the Performance Profiling & Benchmarking Framework (Story 13)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick
from wavelet_research.profiling.benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    LatencyBenchmark,
    MemoryBenchmark,
    ThroughputBenchmark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick(mid: float, index: int = 0) -> Tick:
    return Tick(
        time=pd.Timestamp("2026-06-01") + pd.Timedelta(seconds=index),
        bid=mid - 0.00005,
        ask=mid + 0.00005,
        mid=mid,
        spread=0.00010,
    )


def _make_ticks(n: int = 500, seed: int = 42) -> list[Tick]:
    rng = np.random.default_rng(seed)
    mid_values = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    return [_make_tick(float(m), i) for i, m in enumerate(mid_values)]


# ---------------------------------------------------------------------------
# Latency Benchmark Tests
# ---------------------------------------------------------------------------


class TestLatencyBenchmark:
    def test_runs(self) -> None:
        counter = [0]

        def func() -> None:
            counter[0] += 1

        bench = LatencyBenchmark("test", func, iterations=10, warmup=2)
        result = bench.run()
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test"
        assert result.iterations == 10
        assert result.mean_time_ms >= 0
        assert counter[0] == 12  # 2 warmup + 10 iterations

    def test_measures_time(self) -> None:
        def slow_func() -> None:
            time.sleep(0.001)

        bench = LatencyBenchmark("slow", slow_func, iterations=5, warmup=1)
        result = bench.run()
        assert result.mean_time_ms >= 0.5  # At least 0.5ms

    def test_throughput_computed(self) -> None:
        def fast_func() -> None:
            pass

        bench = LatencyBenchmark("fast", fast_func, iterations=50, warmup=5)
        result = bench.run()
        assert result.throughput_per_sec > 0

    def test_std_computed(self) -> None:
        def func() -> None:
            pass

        bench = LatencyBenchmark("var", func, iterations=20, warmup=2)
        result = bench.run()
        assert result.std_time_ms >= 0

    def test_min_max(self) -> None:
        def func() -> None:
            pass

        bench = LatencyBenchmark("mm", func, iterations=10, warmup=2)
        result = bench.run()
        assert result.min_time_ms <= result.mean_time_ms
        assert result.max_time_ms >= result.mean_time_ms


# ---------------------------------------------------------------------------
# Memory Benchmark Tests
# ---------------------------------------------------------------------------


class TestMemoryBenchmark:
    def test_runs(self) -> None:
        def func() -> None:
            _data = [0] * 10000

        bench = MemoryBenchmark("mem_test", func)
        result = bench.run()
        assert result.name == "mem_test"
        assert result.peak_memory_kb > 0
        assert result.iterations == 1

    def test_measures_allocation(self) -> None:
        def big_alloc() -> None:
            _data = np.zeros(100000)

        bench = MemoryBenchmark("big", big_alloc)
        result = bench.run()
        # 100k float64 = ~800KB
        assert result.peak_memory_kb > 50


# ---------------------------------------------------------------------------
# Throughput Benchmark Tests
# ---------------------------------------------------------------------------


class TestThroughputBenchmark:
    def test_runs(self) -> None:
        ticks = _make_ticks(100)
        engine = WaveletEngine(WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        ))

        def process() -> None:
            for tick in ticks:
                engine.update(tick)

        bench = ThroughputBenchmark("wavelet_100", process, n_items=100)
        result = bench.run()
        assert result.name == "wavelet_100"
        assert result.throughput_per_sec > 0

    def test_throughput_scales(self) -> None:
        ticks_small = _make_ticks(50)
        ticks_large = _make_ticks(200)

        engine1 = WaveletEngine(WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        ))
        engine2 = WaveletEngine(WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        ))

        def small() -> None:
            for t in ticks_small:
                engine1.update(t)

        def large() -> None:
            for t in ticks_large:
                engine2.update(t)

        r1 = ThroughputBenchmark("small", small, 50).run()
        r2 = ThroughputBenchmark("large", large, 200).run()
        # Both should report positive throughput
        assert r1.throughput_per_sec > 0
        assert r2.throughput_per_sec > 0


# ---------------------------------------------------------------------------
# Benchmark Suite Tests
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def test_creates(self) -> None:
        suite = BenchmarkSuite("test_suite")
        assert suite.name == "test_suite"
        assert len(suite.results) == 0

    def test_add_result(self) -> None:
        suite = BenchmarkSuite("s")
        result = BenchmarkResult(
            name="b1", mean_time_ms=5.0, std_time_ms=0.5,
            min_time_ms=4.0, max_time_ms=6.0,
            iterations=10, throughput_per_sec=200.0,
            peak_memory_kb=100.0,
        )
        suite.add_result(result)
        assert len(suite.results) == 1

    def test_no_regression(self) -> None:
        suite = BenchmarkSuite("s")
        suite.set_baseline("b1", 10.0)
        result = BenchmarkResult(
            name="b1", mean_time_ms=10.0, std_time_ms=0.5,
            min_time_ms=9.0, max_time_ms=11.0,
            iterations=10, throughput_per_sec=100.0,
            peak_memory_kb=0.0,
        )
        suite.add_result(result)
        regressions = suite.check_regressions(tolerance=0.2)
        assert len(regressions) == 0

    def test_regression_detected(self) -> None:
        suite = BenchmarkSuite("s")
        suite.set_baseline("b1", 10.0)
        result = BenchmarkResult(
            name="b1", mean_time_ms=15.0, std_time_ms=0.5,
            min_time_ms=14.0, max_time_ms=16.0,
            iterations=10, throughput_per_sec=66.0,
            peak_memory_kb=0.0,
        )
        suite.add_result(result)
        regressions = suite.check_regressions(tolerance=0.2)
        assert len(regressions) == 1
        assert "b1" in regressions[0]

    def test_within_tolerance(self) -> None:
        suite = BenchmarkSuite("s")
        suite.set_baseline("b1", 10.0)
        result = BenchmarkResult(
            name="b1", mean_time_ms=11.5, std_time_ms=0.5,
            min_time_ms=10.0, max_time_ms=12.0,
            iterations=10, throughput_per_sec=87.0,
            peak_memory_kb=0.0,
        )
        suite.add_result(result)
        regressions = suite.check_regressions(tolerance=0.2)
        assert len(regressions) == 0


# ---------------------------------------------------------------------------
# Integration: Wavelet Engine Profiling
# ---------------------------------------------------------------------------


class TestWaveletEngineProfiling:
    def test_latency_benchmark(self) -> None:
        engine = WaveletEngine(WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        ))
        ticks = _make_ticks(300)
        idx = [0]

        def process_one() -> None:
            if idx[0] < len(ticks):
                engine.update(ticks[idx[0]])
                idx[0] += 1

        bench = LatencyBenchmark("wavelet_tick", process_one, iterations=50, warmup=5)
        result = bench.run()
        assert result.mean_time_ms < 100  # Should be fast

    def test_memory_benchmark(self) -> None:
        def full_run() -> None:
            engine = WaveletEngine(WaveletEngineConfig(
                wavelet="haar", window=256, level=2
            ))
            ticks = _make_ticks(500)
            for t in ticks:
                engine.update(t)

        bench = MemoryBenchmark("wavelet_500", full_run)
        result = bench.run()
        assert result.peak_memory_kb > 0
        assert result.peak_memory_kb < 50000  # < 50MB
