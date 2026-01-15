"""
Model Optimization and Benchmarking Module

Provides tools for:
- Inference latency benchmarking
- Model profiling and throughput analysis
- INT8 quantization support
- Memory usage analysis
- Batch inference optimization
- Learning rate warmup scheduling
"""

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from enum import Enum
import gc
import sys


class DeviceType(Enum):
    """Target device types for optimization."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class BenchmarkResult:
    """Result of inference benchmark."""
    model_name: str
    device: str
    batch_size: int
    input_shape: Tuple[int, ...]
    warmup_runs: int
    benchmark_runs: int

    # Timing metrics (in milliseconds)
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput
    throughput_samples_per_sec: float

    # Memory
    peak_memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "input_shape": self.input_shape,
            "warmup_runs": self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Benchmark Results: {self.model_name} ===",
            f"Device: {self.device}",
            f"Batch size: {self.batch_size}",
            f"Input shape: {self.input_shape}",
            "",
            "Latency (ms):",
            f"  Mean: {self.mean_latency_ms:.2f} ± {self.std_latency_ms:.2f}",
            f"  Min:  {self.min_latency_ms:.2f}",
            f"  Max:  {self.max_latency_ms:.2f}",
            f"  P50:  {self.p50_latency_ms:.2f}",
            f"  P95:  {self.p95_latency_ms:.2f}",
            f"  P99:  {self.p99_latency_ms:.2f}",
            "",
            f"Throughput: {self.throughput_samples_per_sec:.1f} samples/sec",
        ]
        if self.peak_memory_mb is not None:
            lines.append(f"Peak Memory: {self.peak_memory_mb:.1f} MB")
        return "\n".join(lines)


@dataclass
class ModelProfile:
    """Model profiling information."""
    model_name: str
    total_params: int
    trainable_params: int
    total_size_mb: float

    # Per-layer info
    layer_params: Dict[str, int] = field(default_factory=dict)
    layer_types: Dict[str, int] = field(default_factory=dict)

    # FLOPS estimate (if computed)
    estimated_flops: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "total_size_mb": self.total_size_mb,
            "layer_params": self.layer_params,
            "layer_types": self.layer_types,
            "estimated_flops": self.estimated_flops,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Model Profile: {self.model_name} ===",
            f"Total parameters: {self.total_params:,}",
            f"Trainable parameters: {self.trainable_params:,}",
            f"Model size: {self.total_size_mb:.2f} MB",
            "",
            "Layer types:",
        ]
        for layer_type, count in sorted(self.layer_types.items()):
            lines.append(f"  {layer_type}: {count}")

        if self.estimated_flops is not None:
            gflops = self.estimated_flops / 1e9
            lines.append(f"\nEstimated FLOPS: {gflops:.2f} GFLOPS")

        return "\n".join(lines)


class ModelBenchmarker:
    """Benchmark model inference performance."""

    def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 100):
        """
        Initialize benchmarker.

        Args:
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Get appropriate device."""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _measure_peak_memory(self, device: torch.device) -> Optional[float]:
        """Measure peak memory usage in MB."""
        if device.type == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return None

    def benchmark(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Benchmark model inference.

        Args:
            model: Model to benchmark
            input_shape: Shape of a single input (excluding batch dimension)
            batch_size: Batch size for inference
            device: Device to benchmark on
            model_name: Optional model name for results

        Returns:
            BenchmarkResult with timing statistics
        """
        dev = self._get_device(device)
        model = model.to(dev)
        model.eval()

        # Create dummy input
        full_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(*full_shape, device=dev)

        # Reset memory stats
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)
                if dev.type == "cuda":
                    torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                if dev.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(dummy_input)

                if dev.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        import numpy as np
        latencies = np.array(latencies)

        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))
        p50_latency = float(np.percentile(latencies, 50))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))

        throughput = (batch_size * 1000) / mean_latency  # samples/sec

        peak_memory = self._measure_peak_memory(dev)

        return BenchmarkResult(
            model_name=model_name or model.__class__.__name__,
            device=str(dev),
            batch_size=batch_size,
            input_shape=input_shape,
            warmup_runs=self.warmup_runs,
            benchmark_runs=self.benchmark_runs,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_samples_per_sec=throughput,
            peak_memory_mb=peak_memory,
        )

    def compare_batch_sizes(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int],
        device: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """
        Compare performance across different batch sizes.

        Args:
            model: Model to benchmark
            input_shape: Shape of a single input
            batch_sizes: List of batch sizes to test
            device: Device to benchmark on

        Returns:
            List of BenchmarkResult for each batch size
        """
        results = []
        for bs in batch_sizes:
            try:
                result = self.benchmark(model, input_shape, bs, device)
                results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at batch size {bs}, stopping")
                    break
                raise
        return results


class ModelProfiler:
    """Profile model architecture and parameters."""

    def profile(self, model: nn.Module, model_name: Optional[str] = None) -> ModelProfile:
        """
        Profile model parameters and architecture.

        Args:
            model: Model to profile
            model_name: Optional model name

        Returns:
            ModelProfile with detailed information
        """
        total_params = 0
        trainable_params = 0
        layer_params: Dict[str, int] = {}
        layer_types: Dict[str, int] = {}

        for name, module in model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue

            module_type = module.__class__.__name__
            layer_types[module_type] = layer_types.get(module_type, 0) + 1

            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_params[name] = params

        for p in model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()

        # Calculate model size in MB (assuming float32)
        total_size_mb = (total_params * 4) / (1024 * 1024)

        return ModelProfile(
            model_name=model_name or model.__class__.__name__,
            total_params=total_params,
            trainable_params=trainable_params,
            total_size_mb=total_size_mb,
            layer_params=layer_params,
            layer_types=layer_types,
        )

    def estimate_flops(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
    ) -> int:
        """
        Estimate FLOPS for forward pass (approximate).

        Args:
            model: Model to analyze
            input_shape: Input shape (excluding batch)
            batch_size: Batch size

        Returns:
            Estimated FLOPS count
        """
        total_flops = 0
        full_shape = (batch_size,) + input_shape

        # Hook to count operations
        flops_dict: Dict[str, int] = {}

        def hook_fn(module, input, output):
            nonlocal total_flops

            if isinstance(module, nn.Linear):
                # FLOPS = 2 * in_features * out_features * batch_size
                in_features = input[0].shape[-1]
                out_features = output.shape[-1]
                batch = input[0].numel() // in_features
                flops = 2 * in_features * out_features * batch
                total_flops += flops

            elif isinstance(module, nn.Conv1d):
                # FLOPS = 2 * in_channels * out_channels * kernel_size * output_size * batch
                out_size = output.shape[-1]
                batch = output.shape[0]
                flops = (2 * module.in_channels * module.out_channels *
                        module.kernel_size[0] * out_size * batch)
                total_flops += flops

            elif isinstance(module, nn.Conv2d):
                out_h, out_w = output.shape[2:]
                batch = output.shape[0]
                flops = (2 * module.in_channels * module.out_channels *
                        module.kernel_size[0] * module.kernel_size[1] *
                        out_h * out_w * batch)
                total_flops += flops

            elif isinstance(module, nn.LSTM):
                # Approximate FLOPS for LSTM
                # 4 gates * 2 (input + hidden) * hidden_size * seq_len * batch
                input_size = module.input_size
                hidden_size = module.hidden_size
                seq_len = input[0].shape[1]
                batch = input[0].shape[0]
                num_dirs = 2 if module.bidirectional else 1
                flops = 8 * (input_size + hidden_size) * hidden_size * seq_len * batch * num_dirs
                total_flops += flops

        # Register hooks
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))

        try:
            # Forward pass
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*full_shape)
                _ = model(dummy_input)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return total_flops


class INT8Quantizer:
    """
    INT8 quantization support using PyTorch's built-in quantization.

    Supports:
    - Dynamic quantization (for LSTM/Transformer models)
    - Static quantization (requires calibration data)
    """

    @staticmethod
    def dynamic_quantize(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.

        Dynamic quantization is suitable for LSTM and Transformer models
        where weights are quantized statically and activations are
        quantized dynamically at runtime.

        Args:
            model: Model to quantize
            dtype: Quantization data type

        Returns:
            Quantized model
        """
        # Dynamic quantization works best with Linear and LSTM layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM},
            dtype=dtype
        )
        return quantized_model

    @staticmethod
    def prepare_static_quantization(
        model: nn.Module,
        backend: str = "fbgemm",
    ) -> nn.Module:
        """
        Prepare model for static quantization.

        After calling this, you need to:
        1. Run calibration data through the model
        2. Call convert_static_quantization()

        Args:
            model: Model to prepare
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)

        Returns:
            Prepared model
        """
        model.eval()

        # Set quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig(backend)

        # Fuse modules where possible
        # Note: This is model-specific, may need customization

        # Prepare for quantization
        prepared_model = torch.quantization.prepare(model)

        return prepared_model

    @staticmethod
    def convert_static_quantization(model: nn.Module) -> nn.Module:
        """
        Convert prepared model to quantized model.

        Call this after running calibration data through the prepared model.

        Args:
            model: Prepared model after calibration

        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.convert(model)
        return quantized_model

    @staticmethod
    def get_quantized_size(model: nn.Module) -> float:
        """
        Estimate size of quantized model in MB.

        Args:
            model: Model to analyze

        Returns:
            Estimated size in MB
        """
        total_bytes = 0
        for name, param in model.named_parameters():
            if hasattr(param, "q_per_channel_scales"):
                # Quantized parameter (int8)
                total_bytes += param.numel() * 1
            else:
                # Non-quantized parameter (float32)
                total_bytes += param.numel() * 4

        return total_bytes / (1024 * 1024)


@dataclass
class MemoryAnalysis:
    """Memory usage analysis result."""
    model_size_mb: float
    activation_size_mb: float
    gradient_size_mb: float
    optimizer_state_mb: float
    total_training_mb: float
    total_inference_mb: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""=== Memory Analysis ===
Model parameters: {self.model_size_mb:.2f} MB
Activations: {self.activation_size_mb:.2f} MB
Gradients: {self.gradient_size_mb:.2f} MB
Optimizer state: {self.optimizer_state_mb:.2f} MB

Total for inference: {self.total_inference_mb:.2f} MB
Total for training: {self.total_training_mb:.2f} MB"""


class MemoryAnalyzer:
    """Analyze memory usage of models."""

    def analyze(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        optimizer_type: str = "adam",
    ) -> MemoryAnalysis:
        """
        Analyze memory requirements for model.

        Args:
            model: Model to analyze
            input_shape: Input shape (excluding batch)
            batch_size: Batch size
            optimizer_type: Optimizer type ('sgd', 'adam')

        Returns:
            MemoryAnalysis with detailed breakdown
        """
        # Model size
        model_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (model_params * 4) / (1024 * 1024)  # float32

        # Gradient size (same as model size)
        gradient_size_mb = model_size_mb

        # Optimizer state size
        if optimizer_type.lower() == "adam":
            # Adam stores m and v for each parameter
            optimizer_state_mb = model_size_mb * 2
        elif optimizer_type.lower() == "sgd":
            # SGD with momentum stores velocity
            optimizer_state_mb = model_size_mb
        else:
            optimizer_state_mb = model_size_mb

        # Activation size (estimate by running forward pass)
        activation_size_mb = self._estimate_activations(model, input_shape, batch_size)

        total_inference_mb = model_size_mb + activation_size_mb
        total_training_mb = (
            model_size_mb +
            activation_size_mb +
            gradient_size_mb +
            optimizer_state_mb
        )

        return MemoryAnalysis(
            model_size_mb=model_size_mb,
            activation_size_mb=activation_size_mb,
            gradient_size_mb=gradient_size_mb,
            optimizer_state_mb=optimizer_state_mb,
            total_training_mb=total_training_mb,
            total_inference_mb=total_inference_mb,
        )

    def _estimate_activations(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int,
    ) -> float:
        """Estimate activation memory in MB."""
        total_elements = 0

        def hook_fn(module, input, output):
            nonlocal total_elements
            if isinstance(output, torch.Tensor):
                total_elements += output.numel()
            elif isinstance(output, tuple):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        total_elements += o.numel()

        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))

        try:
            model.eval()
            with torch.no_grad():
                full_shape = (batch_size,) + input_shape
                dummy_input = torch.randn(*full_shape)
                _ = model(dummy_input)
        finally:
            for hook in hooks:
                hook.remove()

        # Float32 = 4 bytes
        return (total_elements * 4) / (1024 * 1024)


class GradientAccumulator:
    """
    Gradient accumulation for training with larger effective batch sizes.

    Usage:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for batch in dataloader:
            loss = model(batch)
            loss = accumulator.scale_loss(loss)
            loss.backward()

            if accumulator.should_step():
                optimizer.step()
                optimizer.zero_grad()

            accumulator.step()
    """

    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")

        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for gradient accumulation.

        Args:
            loss: Original loss value

        Returns:
            Scaled loss
        """
        return loss / self.accumulation_steps

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return (self.current_step + 1) % self.accumulation_steps == 0

    def step(self):
        """Increment step counter."""
        self.current_step += 1

    def reset(self):
        """Reset step counter."""
        self.current_step = 0

    @property
    def effective_batch_size(self) -> Callable[[int], int]:
        """Get effective batch size calculator."""
        return lambda bs: bs * self.accumulation_steps


class WarmupScheduler:
    """
    Learning rate warmup scheduler.

    Implements linear warmup followed by decay strategy.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.0,
        decay_type: str = "linear",
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate after decay
            warmup_start_lr: Starting LR for warmup
            decay_type: Decay type ('linear', 'cosine')
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.decay_type = decay_type

        # Store initial learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            return [
                self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Decay phase
            decay_steps = self.total_steps - self.warmup_steps
            decay_progress = (self.current_step - self.warmup_steps) / max(decay_steps, 1)

            if self.decay_type == "cosine":
                import math
                decay_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:  # linear
                decay_factor = 1 - decay_progress

            return [
                self.min_lr + decay_factor * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]

    def step(self):
        """Update learning rate."""
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self.current_step += 1

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state."""
        self.current_step = state_dict["current_step"]
        self.base_lrs = state_dict["base_lrs"]


class BatchOptimizer:
    """
    Optimize batch processing for inference.

    Provides utilities for efficient batch inference.
    """

    @staticmethod
    def find_optimal_batch_size(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        max_batch_size: int = 256,
        device: Optional[str] = None,
    ) -> int:
        """
        Find optimal batch size for given model and device.

        Uses binary search to find maximum batch size that fits in memory.

        Args:
            model: Model to test
            input_shape: Input shape (excluding batch)
            max_batch_size: Maximum batch size to try
            device: Device to test on

        Returns:
            Optimal batch size
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        dev = torch.device(device)
        model = model.to(dev)
        model.eval()

        # Binary search
        low, high = 1, max_batch_size
        optimal = 1

        while low <= high:
            mid = (low + high) // 2
            try:
                # Clear memory
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

                # Test batch size
                full_shape = (mid,) + input_shape
                test_input = torch.randn(*full_shape, device=dev)

                with torch.no_grad():
                    _ = model(test_input)

                if dev.type == "cuda":
                    torch.cuda.synchronize()

                # Success - try larger
                optimal = mid
                low = mid + 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                else:
                    raise
            finally:
                # Cleanup
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

        return optimal

    @staticmethod
    def batch_inference(
        model: nn.Module,
        data: torch.Tensor,
        batch_size: int,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Perform batched inference on large dataset.

        Args:
            model: Model for inference
            data: Input data tensor
            batch_size: Batch size for inference
            device: Device to use

        Returns:
            Output tensor
        """
        if device is None:
            device = "cpu"

        dev = torch.device(device)
        model = model.to(dev)
        model.eval()

        outputs = []
        n_samples = data.shape[0]

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = data[i:i + batch_size].to(dev)
                output = model(batch)
                outputs.append(output.cpu())

        return torch.cat(outputs, dim=0)
