"""
Tests for gesture recognition module.

Tests the 3D CNN model, dataset classes, and preprocessing pipeline.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

from wavira.models.gesture_recognizer import (
    GestureRecognizer3DCNN,
    GestureRecognizerLite,
    DualESP32GestureRecognizer,
    GestureRecognizerConfig,
    create_gesture_model,
    DEFAULT_GESTURE_LABELS,
)
from wavira.data.gesture_dataset import (
    GestureDataset,
    DualDeviceGestureDataset,
    SyntheticGestureDataset,
)
from wavira.data.gesture_preprocessing import (
    butterworth_lowpass_filter,
    butterworth_bandpass_filter,
    preprocess_gesture_csi,
    GesturePreprocessor,
    GesturePreprocessConfig,
    normalize_csi,
)


class TestGestureRecognizer3DCNN:
    """Tests for GestureRecognizer3DCNN model."""

    @pytest.fixture
    def config(self):
        return GestureRecognizerConfig(
            n_gestures=8,
            n_subcarriers=114,
            n_routes=3,
            n_frames=32,
        )

    @pytest.fixture
    def model(self, config):
        return GestureRecognizer3DCNN(config)

    def test_model_creation(self, model, config):
        """Test model can be created with config."""
        assert model.n_gestures == config.n_gestures
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self, model):
        """Test forward pass with correct input shape."""
        batch_size = 4
        n_routes = 3
        n_subcarriers = 114
        n_frames = 32

        # Input shape: (batch, n_routes, n_subcarriers, n_frames)
        x = torch.randn(batch_size, n_routes, n_subcarriers, n_frames)
        output = model(x)

        assert output.shape == (batch_size, 8)

    def test_forward_pass_5d_input(self, model):
        """Test forward pass with 5D input."""
        batch_size = 4
        n_frames = 32
        n_routes = 3
        n_subcarriers = 114

        # Input shape: (batch, 1, n_frames, n_routes, n_subcarriers)
        x = torch.randn(batch_size, 1, n_frames, n_routes, n_subcarriers)
        output = model(x)

        assert output.shape == (batch_size, 8)

    def test_predict_method(self, model):
        """Test predict method returns predictions and confidence."""
        x = torch.randn(4, 3, 114, 32)
        predictions, confidence = model.predict(x)

        assert predictions.shape == (4,)
        assert confidence.shape == (4,)
        assert (predictions >= 0).all() and (predictions < 8).all()
        assert (confidence >= 0).all() and (confidence <= 1).all()


class TestGestureRecognizerLite:
    """Tests for lightweight gesture recognizer."""

    def test_model_creation(self):
        """Test lite model creation."""
        model = GestureRecognizerLite()
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass."""
        model = GestureRecognizerLite()
        x = torch.randn(4, 3, 114, 32)
        output = model(x)

        assert output.shape == (4, 8)

    def test_fewer_parameters(self):
        """Test that lite model has fewer parameters."""
        standard = GestureRecognizer3DCNN()
        lite = GestureRecognizerLite()

        standard_params = sum(p.numel() for p in standard.parameters())
        lite_params = sum(p.numel() for p in lite.parameters())

        assert lite_params < standard_params


class TestDualESP32GestureRecognizer:
    """Tests for dual-device gesture recognizer."""

    @pytest.fixture
    def model(self):
        return DualESP32GestureRecognizer()

    def test_forward_dual_input(self, model):
        """Test forward with two device inputs."""
        x1 = torch.randn(4, 3, 114, 32)
        x2 = torch.randn(4, 3, 114, 32)
        output = model(x1, x2)

        assert output.shape == (4, 8)

    def test_forward_single_input(self, model):
        """Test forward with single device input."""
        x1 = torch.randn(4, 3, 114, 32)
        output = model(x1, None)

        assert output.shape == (4, 8)

    def test_fusion_methods(self):
        """Test different fusion methods."""
        for method in ['concat', 'add']:
            model = DualESP32GestureRecognizer(fusion_method=method)
            x1 = torch.randn(2, 3, 114, 32)
            x2 = torch.randn(2, 3, 114, 32)
            output = model(x1, x2)
            assert output.shape == (2, 8)


class TestCreateGestureModel:
    """Tests for model factory function."""

    def test_create_standard_model(self):
        """Test creating standard model."""
        model = create_gesture_model(model_type="standard")
        assert isinstance(model, GestureRecognizer3DCNN)

    def test_create_lite_model(self):
        """Test creating lite model."""
        model = create_gesture_model(model_type="lite")
        assert isinstance(model, GestureRecognizerLite)

    def test_create_dual_model(self):
        """Test creating dual model."""
        model = create_gesture_model(model_type="dual")
        assert isinstance(model, DualESP32GestureRecognizer)

    def test_custom_n_gestures(self):
        """Test creating model with custom number of gestures."""
        model = create_gesture_model(n_gestures=4)
        x = torch.randn(2, 3, 114, 32)
        output = model(x)
        assert output.shape == (2, 4)


class TestSyntheticGestureDataset:
    """Tests for synthetic gesture dataset."""

    @pytest.fixture
    def dataset(self):
        return SyntheticGestureDataset(
            n_samples_per_gesture=10,
            n_frames=32,
            n_routes=3,
            n_subcarriers=114,
        )

    def test_dataset_length(self, dataset):
        """Test dataset has correct length."""
        assert len(dataset) == 8 * 10  # 8 gestures * 10 samples

    def test_sample_shape(self, dataset):
        """Test sample shape."""
        csi, label = dataset[0]
        assert csi.shape == (3, 114, 32)
        assert isinstance(label, int)
        assert 0 <= label < 8

    def test_all_labels_present(self, dataset):
        """Test all gesture labels are represented."""
        labels = set()
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.add(label)

        assert labels == set(range(8))


class TestGestureDataset:
    """Tests for GestureDataset with file loading."""

    def test_dataset_from_samples(self):
        """Test adding samples programmatically."""
        dataset = GestureDataset(n_frames=32, frame_stride=8)

        # Add samples
        csi = np.random.randn(3, 114, 100).astype(np.float32)
        dataset.add_samples(csi, "zoom_out")

        assert len(dataset) > 0

    def test_window_extraction(self):
        """Test sliding window extraction."""
        dataset = GestureDataset(n_frames=32, frame_stride=8)

        # 100 frames should produce (100 - 32) / 8 + 1 = 9 windows
        csi = np.random.randn(3, 114, 100).astype(np.float32)
        dataset.add_samples(csi, 0)

        expected_windows = (100 - 32) // 8 + 1
        assert len(dataset) == expected_windows


class TestButterworthFilter:
    """Tests for Butterworth filter preprocessing."""

    def test_lowpass_filter_shape(self):
        """Test filter preserves shape."""
        data = np.random.randn(3, 114, 100)
        filtered = butterworth_lowpass_filter(data)
        assert filtered.shape == data.shape

    def test_lowpass_filter_reduces_noise(self):
        """Test filter reduces high-frequency noise."""
        # Create signal with noise
        t = np.linspace(0, 1, 100)
        clean = np.sin(2 * np.pi * 5 * t)  # 5 Hz signal
        noise = 0.5 * np.sin(2 * np.pi * 40 * t)  # 40 Hz noise
        noisy = clean + noise

        # Expand to 3D
        data = np.tile(noisy, (3, 114, 1))
        filtered = butterworth_lowpass_filter(data, cutoff=20.0)

        # Filtered should be closer to clean signal
        assert np.std(filtered[0, 0] - clean) < np.std(noisy - clean)

    def test_bandpass_filter(self):
        """Test bandpass filter."""
        data = np.random.randn(3, 114, 100)
        filtered = butterworth_bandpass_filter(data, low_cutoff=1.0, high_cutoff=20.0)
        assert filtered.shape == data.shape


class TestGesturePreprocessor:
    """Tests for GesturePreprocessor callable."""

    def test_preprocessor_callable(self):
        """Test preprocessor can be called on data."""
        preprocessor = GesturePreprocessor()
        data = np.random.randn(3, 114, 100)
        processed = preprocessor(data)

        assert processed.shape == data.shape
        assert processed.dtype == np.float32

    def test_preprocessor_with_custom_config(self):
        """Test preprocessor with custom configuration."""
        preprocessor = GesturePreprocessor(
            butter_cutoff=15.0,
            sampling_rate=50.0,
            normalize=True,
        )
        data = np.random.randn(3, 114, 100)
        processed = preprocessor(data)

        assert processed.shape == data.shape

    def test_preprocessor_normalization(self):
        """Test that preprocessor normalizes data."""
        preprocessor = GesturePreprocessor(normalize=True)
        data = np.random.randn(3, 114, 100) * 100 + 50  # Scale and shift

        processed = preprocessor(data)

        # Should be approximately normalized
        assert np.abs(processed.mean()) < 1.0
        assert 0.5 < processed.std() < 2.0


class TestNormalizeCsi:
    """Tests for CSI normalization."""

    def test_normalize_zero_mean(self):
        """Test normalization produces zero mean."""
        data = np.random.randn(3, 114, 100) * 10 + 5
        normalized = normalize_csi(data, per_subcarrier=False)

        assert np.abs(normalized.mean()) < 0.1

    def test_normalize_per_subcarrier(self):
        """Test per-subcarrier normalization."""
        data = np.random.randn(3, 114, 100)
        normalized = normalize_csi(data, per_subcarrier=True)

        # Each time series should be normalized
        for r in range(3):
            for s in range(114):
                series = normalized[r, s]
                assert np.abs(series.mean()) < 0.1


class TestPreprocessGestureCsi:
    """Tests for full preprocessing pipeline."""

    def test_pipeline_output_shape(self):
        """Test pipeline preserves shape."""
        data = np.random.randn(3, 114, 100)
        processed = preprocess_gesture_csi(data)

        assert processed.shape == data.shape

    def test_pipeline_handles_complex(self):
        """Test pipeline handles complex input."""
        real = np.random.randn(3, 114, 100)
        imag = np.random.randn(3, 114, 100)
        data = real + 1j * imag

        config = GesturePreprocessConfig(extract_amplitude=True)
        processed = preprocess_gesture_csi(data, config)

        assert processed.shape == (3, 114, 100)
        assert not np.iscomplexobj(processed)


class TestDefaultGestureLabels:
    """Tests for default gesture labels."""

    def test_eight_gestures(self):
        """Test there are 8 default gestures."""
        assert len(DEFAULT_GESTURE_LABELS) == 8

    def test_expected_gestures(self):
        """Test expected gesture names are present."""
        expected = ["zoom_out", "zoom_in", "circle_left", "circle_right",
                    "swipe_left", "swipe_right", "flip_up", "flip_down"]
        assert DEFAULT_GESTURE_LABELS == expected


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_training_loop_synthetic(self):
        """Test a mini training loop with synthetic data."""
        # Create model
        model = create_gesture_model(model_type="lite", n_gestures=4)

        # Create dataset
        dataset = SyntheticGestureDataset(
            n_samples_per_gesture=20,
            n_frames=32,
            gesture_labels=["zoom_out", "zoom_in", "swipe_left", "swipe_right"],
        )

        # Create dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # One training step
        model.train()
        for csi, labels in loader:
            optimizer.zero_grad()
            outputs = model(csi)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break  # Just one batch

        # Verify model still works
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 114, 32)
            output = model(test_input)
            assert output.shape == (1, 4)

    def test_preprocessing_to_model(self):
        """Test data flows correctly from preprocessing to model."""
        preprocessor = GesturePreprocessor()
        model = create_gesture_model(model_type="lite")

        # Raw CSI data
        raw_csi = np.random.randn(3, 114, 32).astype(np.float32)

        # Preprocess
        processed = preprocessor(raw_csi)

        # Convert to tensor and add batch dim
        tensor = torch.from_numpy(processed).unsqueeze(0)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(tensor)

        assert output.shape == (1, 8)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
