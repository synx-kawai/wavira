"""Tests for model optimization module."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from wavira.models.optimization import (
    BenchmarkResult,
    ModelProfile,
    ModelBenchmarker,
    ModelProfiler,
    INT8Quantizer,
    MemoryAnalysis,
    MemoryAnalyzer,
    GradientAccumulator,
    WarmupScheduler,
    BatchOptimizer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class LSTMModel(nn.Module):
    """LSTM model for testing."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.linear(output[:, -1, :])


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_to_dict(self):
        result = BenchmarkResult(
            model_name="test",
            device="cpu",
            batch_size=1,
            input_shape=(64,),
            warmup_runs=10,
            benchmark_runs=100,
            mean_latency_ms=1.0,
            std_latency_ms=0.1,
            min_latency_ms=0.8,
            max_latency_ms=1.5,
            p50_latency_ms=1.0,
            p95_latency_ms=1.2,
            p99_latency_ms=1.4,
            throughput_samples_per_sec=1000.0,
        )
        d = result.to_dict()
        assert d["model_name"] == "test"
        assert d["mean_latency_ms"] == 1.0
        assert d["throughput_samples_per_sec"] == 1000.0

    def test_summary(self):
        result = BenchmarkResult(
            model_name="test",
            device="cpu",
            batch_size=1,
            input_shape=(64,),
            warmup_runs=10,
            benchmark_runs=100,
            mean_latency_ms=1.0,
            std_latency_ms=0.1,
            min_latency_ms=0.8,
            max_latency_ms=1.5,
            p50_latency_ms=1.0,
            p95_latency_ms=1.2,
            p99_latency_ms=1.4,
            throughput_samples_per_sec=1000.0,
            peak_memory_mb=50.0,
        )
        summary = result.summary()
        assert "test" in summary
        assert "cpu" in summary
        assert "Latency" in summary
        assert "Throughput" in summary
        assert "Memory" in summary


class TestModelProfile:
    """Tests for ModelProfile."""

    def test_to_dict(self):
        profile = ModelProfile(
            model_name="test",
            total_params=1000,
            trainable_params=800,
            total_size_mb=0.5,
        )
        d = profile.to_dict()
        assert d["model_name"] == "test"
        assert d["total_params"] == 1000
        assert d["trainable_params"] == 800

    def test_summary(self):
        profile = ModelProfile(
            model_name="test",
            total_params=1000,
            trainable_params=800,
            total_size_mb=0.5,
            layer_types={"Linear": 2, "ReLU": 1},
        )
        summary = profile.summary()
        assert "test" in summary
        assert "1,000" in summary
        assert "Linear" in summary


class TestModelBenchmarker:
    """Tests for ModelBenchmarker."""

    def test_benchmark_simple_model(self):
        model = SimpleModel()
        benchmarker = ModelBenchmarker(warmup_runs=2, benchmark_runs=10)
        # Explicitly use CPU to avoid device mismatch
        result = benchmarker.benchmark(model, input_shape=(64,), batch_size=1, device="cpu")

        assert result.model_name == "SimpleModel"
        assert result.device == "cpu"
        assert result.batch_size == 1
        assert result.mean_latency_ms > 0
        assert result.throughput_samples_per_sec > 0

    def test_benchmark_different_batch_sizes(self):
        model = SimpleModel()
        benchmarker = ModelBenchmarker(warmup_runs=2, benchmark_runs=5)

        result1 = benchmarker.benchmark(model, input_shape=(64,), batch_size=1, device="cpu")
        result2 = benchmarker.benchmark(model, input_shape=(64,), batch_size=8, device="cpu")

        # Larger batch should have higher throughput
        assert result2.throughput_samples_per_sec > result1.throughput_samples_per_sec * 0.5

    def test_benchmark_lstm_model(self):
        model = LSTMModel()
        benchmarker = ModelBenchmarker(warmup_runs=2, benchmark_runs=5)
        result = benchmarker.benchmark(model, input_shape=(100, 64), batch_size=1, device="cpu")

        assert result.model_name == "LSTMModel"
        assert result.input_shape == (100, 64)

    def test_compare_batch_sizes(self):
        model = SimpleModel()
        benchmarker = ModelBenchmarker(warmup_runs=2, benchmark_runs=5)
        results = benchmarker.compare_batch_sizes(
            model,
            input_shape=(64,),
            batch_sizes=[1, 4, 8],
            device="cpu",
        )

        assert len(results) == 3
        assert results[0].batch_size == 1
        assert results[1].batch_size == 4
        assert results[2].batch_size == 8


class TestModelProfiler:
    """Tests for ModelProfiler."""

    def test_profile_simple_model(self):
        model = SimpleModel(input_dim=64, hidden_dim=128, output_dim=10)
        profiler = ModelProfiler()
        profile = profiler.profile(model)

        # Expected params: 64*128 + 128 + 128*10 + 10 = 8192 + 128 + 1280 + 10 = 9610
        assert profile.total_params == 9610
        assert profile.trainable_params == 9610
        assert profile.total_size_mb > 0
        assert "Linear" in profile.layer_types
        assert profile.layer_types["Linear"] == 2

    def test_profile_model_with_frozen_params(self):
        model = SimpleModel()
        # Freeze first layer
        for param in model.linear1.parameters():
            param.requires_grad = False

        profiler = ModelProfiler()
        profile = profiler.profile(model)

        assert profile.trainable_params < profile.total_params

    def test_estimate_flops(self):
        model = SimpleModel(input_dim=64, hidden_dim=128, output_dim=10)
        profiler = ModelProfiler()
        flops = profiler.estimate_flops(model, input_shape=(64,), batch_size=1)

        assert flops > 0
        # Expected: 2 * 64 * 128 * 1 + 2 * 128 * 10 * 1 = 16384 + 2560 = 18944
        assert flops == pytest.approx(18944, rel=0.01)


class TestINT8Quantizer:
    """Tests for INT8Quantizer."""

    @pytest.mark.skipif(
        not torch.backends.quantized.engine in ['fbgemm', 'qnnpack'],
        reason="Quantization not supported on this platform"
    )
    def test_dynamic_quantize(self):
        model = SimpleModel()
        original_size = sum(p.numel() for p in model.parameters())

        quantized = INT8Quantizer.dynamic_quantize(model)

        # Model should still work
        test_input = torch.randn(1, 64)
        with torch.no_grad():
            output = quantized(test_input)

        assert output.shape == (1, 10)

    @pytest.mark.skipif(
        not torch.backends.quantized.engine in ['fbgemm', 'qnnpack'],
        reason="Quantization not supported on this platform"
    )
    def test_dynamic_quantize_lstm(self):
        model = LSTMModel()
        quantized = INT8Quantizer.dynamic_quantize(model)

        # Model should still work
        test_input = torch.randn(1, 100, 64)
        with torch.no_grad():
            output = quantized(test_input)

        assert output.shape == (1, 10)

    def test_get_quantized_size(self):
        model = SimpleModel()
        size = INT8Quantizer.get_quantized_size(model)

        # Non-quantized model should have full float32 size
        expected_params = 9610
        expected_size_mb = (expected_params * 4) / (1024 * 1024)
        # Allow some tolerance for different implementations
        assert size > 0
        assert size <= expected_size_mb * 1.5  # Should be at most 1.5x expected


class TestMemoryAnalyzer:
    """Tests for MemoryAnalyzer."""

    def test_analyze_simple_model(self):
        model = SimpleModel()
        analyzer = MemoryAnalyzer()
        analysis = analyzer.analyze(model, input_shape=(64,), batch_size=1)

        assert analysis.model_size_mb > 0
        assert analysis.activation_size_mb > 0
        assert analysis.gradient_size_mb > 0
        assert analysis.optimizer_state_mb > 0
        assert analysis.total_training_mb > analysis.total_inference_mb

    def test_analyze_with_adam(self):
        model = SimpleModel()
        analyzer = MemoryAnalyzer()

        adam_analysis = analyzer.analyze(model, input_shape=(64,), optimizer_type="adam")
        sgd_analysis = analyzer.analyze(model, input_shape=(64,), optimizer_type="sgd")

        # Adam uses more memory for optimizer state
        assert adam_analysis.optimizer_state_mb > sgd_analysis.optimizer_state_mb

    def test_analysis_summary(self):
        model = SimpleModel()
        analyzer = MemoryAnalyzer()
        analysis = analyzer.analyze(model, input_shape=(64,))
        summary = analysis.summary()

        assert "Memory Analysis" in summary
        assert "Model parameters" in summary
        assert "Activations" in summary


class TestGradientAccumulator:
    """Tests for GradientAccumulator."""

    def test_scale_loss(self):
        accumulator = GradientAccumulator(accumulation_steps=4)
        loss = torch.tensor(8.0)
        scaled = accumulator.scale_loss(loss)
        assert scaled.item() == 2.0

    def test_should_step(self):
        accumulator = GradientAccumulator(accumulation_steps=4)

        assert not accumulator.should_step()  # Step 0
        accumulator.step()
        assert not accumulator.should_step()  # Step 1
        accumulator.step()
        assert not accumulator.should_step()  # Step 2
        accumulator.step()
        assert accumulator.should_step()  # Step 3

    def test_reset(self):
        accumulator = GradientAccumulator(accumulation_steps=4)
        accumulator.step()
        accumulator.step()
        accumulator.reset()
        assert accumulator.current_step == 0

    def test_effective_batch_size(self):
        accumulator = GradientAccumulator(accumulation_steps=4)
        calc = accumulator.effective_batch_size
        assert calc(8) == 32
        assert calc(16) == 64

    def test_invalid_steps(self):
        with pytest.raises(ValueError):
            GradientAccumulator(accumulation_steps=0)


class TestWarmupScheduler:
    """Tests for WarmupScheduler."""

    def test_warmup_phase(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            warmup_start_lr=0.0,
        )

        # At step 0
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.0, abs=1e-6)

        # At step 50 (halfway through warmup)
        scheduler.current_step = 50
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.0005, abs=1e-6)

        # At step 100 (end of warmup)
        scheduler.current_step = 100
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.001, abs=1e-6)

    def test_decay_phase_linear(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
            decay_type="linear",
        )

        # At step 550 (halfway through decay)
        scheduler.current_step = 550
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.0005, abs=1e-5)

        # At step 1000 (end)
        scheduler.current_step = 1000
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.0, abs=1e-6)

    def test_decay_phase_cosine(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
            decay_type="cosine",
        )

        # At step 550 (halfway through decay)
        scheduler.current_step = 550
        lrs = scheduler.get_lr()
        # Cosine at 0.5 progress = 0.5 * (1 + cos(0.5*pi)) = 0.5
        assert lrs[0] == pytest.approx(0.0005, abs=1e-5)

    def test_step_updates_optimizer(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            warmup_start_lr=0.0,
        )

        # At step 0, scheduler.step() first gets LR for step 0, then increments
        # LR at step 0 = warmup_start_lr = 0.0
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)

        # After second step, we get LR for step 1
        # LR at step 1 = 0 + (1/100) * (0.001 - 0) = 0.00001
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00001, rel=0.01)

    def test_state_dict(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        scheduler.step()
        scheduler.step()
        state = scheduler.state_dict()

        assert state["current_step"] == 2
        assert len(state["base_lrs"]) == 1

    def test_load_state_dict(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        scheduler.load_state_dict({"current_step": 50, "base_lrs": [0.001]})
        assert scheduler.current_step == 50


class TestBatchOptimizer:
    """Tests for BatchOptimizer."""

    def test_find_optimal_batch_size(self):
        model = SimpleModel()
        optimal_bs = BatchOptimizer.find_optimal_batch_size(
            model,
            input_shape=(64,),
            max_batch_size=64,
            device="cpu",
        )

        # On CPU, should be able to use max batch size
        assert optimal_bs == 64

    def test_batch_inference(self):
        model = SimpleModel()
        model.eval()

        # Create test data
        data = torch.randn(100, 64)

        # Run batch inference
        output = BatchOptimizer.batch_inference(
            model,
            data,
            batch_size=32,
            device="cpu",
        )

        assert output.shape == (100, 10)

    def test_batch_inference_small_batch(self):
        model = SimpleModel()
        model.eval()

        data = torch.randn(10, 64)

        output = BatchOptimizer.batch_inference(
            model,
            data,
            batch_size=3,
            device="cpu",
        )

        assert output.shape == (10, 10)


class TestIntegration:
    """Integration tests for optimization module."""

    def test_full_optimization_workflow(self):
        # Create model
        model = SimpleModel(input_dim=64, hidden_dim=256, output_dim=10)

        # Profile
        profiler = ModelProfiler()
        profile = profiler.profile(model)
        assert profile.total_params > 0

        # Benchmark (explicitly on CPU)
        benchmarker = ModelBenchmarker(warmup_runs=2, benchmark_runs=5)
        benchmark = benchmarker.benchmark(model, input_shape=(64,), batch_size=1, device="cpu")
        assert benchmark.mean_latency_ms > 0

        # Memory analysis
        analyzer = MemoryAnalyzer()
        memory = analyzer.analyze(model, input_shape=(64,))
        assert memory.total_training_mb > 0

        # Skip quantization test on platforms that don't support it
        if torch.backends.quantized.engine in ['fbgemm', 'qnnpack']:
            quantized = INT8Quantizer.dynamic_quantize(model)
            benchmark_quant = benchmarker.benchmark(
                quantized, input_shape=(64,), batch_size=1, device="cpu"
            )
            assert benchmark_quant.mean_latency_ms > 0

    def test_training_with_gradient_accumulation(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        accumulator = GradientAccumulator(accumulation_steps=4)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            warmup_start_lr=0.0,
        )

        # Simulate training loop
        for step in range(8):
            data = torch.randn(8, 64)
            target = torch.zeros(8, dtype=torch.long)

            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            scaled_loss = accumulator.scale_loss(loss)
            scaled_loss.backward()

            if accumulator.should_step():
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            accumulator.step()

        # Should have done 2 optimizer steps
        assert scheduler.current_step == 2
