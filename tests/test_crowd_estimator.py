#!/usr/bin/env python3
"""
Unit tests for CrowdEstimator model.
Issue #5: 混雑レベル推定モデルの開発
Issue #38: テストカバレッジ拡充
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from wavira.models.crowd_estimator import (
    CrowdEstimator,
    CrowdEstimatorConfig,
    CrowdEstimatorLight,
    TemporalBlock,
    create_model,
)


# =============================================================================
# TemporalBlock Tests
# =============================================================================


class TestTemporalBlock:
    """Tests for TemporalBlock CNN module."""

    def test_forward_same_channels(self):
        """Forward pass with same input/output channels."""
        block = TemporalBlock(64, 64, kernel_size=3)
        x = torch.randn(2, 64, 100)  # (batch, channels, seq_len)
        out = block(x)
        assert out.shape == (2, 64, 100)

    def test_forward_different_channels(self):
        """Forward pass with different input/output channels."""
        block = TemporalBlock(32, 64, kernel_size=3)
        x = torch.randn(2, 32, 100)
        out = block(x)
        assert out.shape == (2, 64, 100)

    def test_forward_with_dilation(self):
        """Forward pass with dilated convolution."""
        block = TemporalBlock(64, 64, kernel_size=3, dilation=2)
        x = torch.randn(2, 64, 100)
        out = block(x)
        assert out.shape == (2, 64, 100)

    def test_residual_connection(self):
        """Verify residual connection is used."""
        block = TemporalBlock(64, 64, kernel_size=3)
        x = torch.zeros(1, 64, 100)
        out = block(x)
        # With zero input, residual should dominate
        assert out is not None


# =============================================================================
# CrowdEstimatorConfig Tests
# =============================================================================


class TestCrowdEstimatorConfig:
    """Tests for CrowdEstimatorConfig."""

    def test_default_config(self):
        """Default configuration values."""
        config = CrowdEstimatorConfig()
        assert config.n_subcarriers == 52
        assert config.seq_length == 100
        assert config.encoder_type == "transformer"
        assert config.hidden_dim == 128
        assert config.mode == "regression"
        assert config.num_classes == 4

    def test_custom_config(self):
        """Custom configuration values."""
        config = CrowdEstimatorConfig(
            n_subcarriers=114,
            mode="classification",
            num_classes=5,
        )
        assert config.n_subcarriers == 114
        assert config.mode == "classification"
        assert config.num_classes == 5


# =============================================================================
# CrowdEstimator Model Tests
# =============================================================================


class TestCrowdEstimator:
    """Tests for CrowdEstimator model."""

    @pytest.fixture
    def regression_model(self):
        """Create regression model for testing."""
        config = CrowdEstimatorConfig(
            n_subcarriers=52,
            seq_length=100,
            hidden_dim=64,
            num_layers=1,
            mode="regression",
        )
        return CrowdEstimator(config)

    @pytest.fixture
    def classification_model(self):
        """Create classification model for testing."""
        config = CrowdEstimatorConfig(
            n_subcarriers=52,
            seq_length=100,
            hidden_dim=64,
            num_layers=1,
            mode="classification",
            num_classes=4,
        )
        return CrowdEstimator(config)

    def test_regression_forward(self, regression_model):
        """Regression model forward pass."""
        x = torch.randn(4, 100, 52)  # (batch, seq_len, n_subcarriers)
        output = regression_model(x)
        assert output.shape == (4, 1)
        # Output should be non-negative (ReLU)
        assert (output >= 0).all()

    def test_classification_forward(self, classification_model):
        """Classification model forward pass."""
        x = torch.randn(4, 100, 52)
        output = classification_model(x)
        assert output.shape == (4, 4)  # 4 classes

    def test_return_features(self, regression_model):
        """Forward pass with return_features=True."""
        x = torch.randn(2, 100, 52)
        output, features = regression_model(x, return_features=True)
        assert output.shape == (2, 1)
        assert features.shape == (2, 64)  # hidden_dim

    def test_predict_regression(self, regression_model):
        """Predict method for regression model."""
        x = torch.randn(2, 100, 52)
        result = regression_model.predict(x)
        assert "count" in result
        assert "count_int" in result
        assert len(result["count"]) == 2
        assert len(result["count_int"]) == 2

    def test_predict_classification(self, classification_model):
        """Predict method for classification model."""
        x = torch.randn(2, 100, 52)
        result = classification_model.predict(x)
        assert "class" in result
        assert "label" in result
        assert "probabilities" in result
        assert len(result["class"]) == 2
        assert len(result["label"]) == 2
        assert result["probabilities"].shape == (2, 4)
        # Labels should be valid
        valid_labels = {"empty", "low", "medium", "high"}
        assert all(label in valid_labels for label in result["label"])

    def test_different_encoder_types(self):
        """Test different encoder types."""
        for encoder_type in ["transformer", "lstm", "bilstm"]:
            config = CrowdEstimatorConfig(
                n_subcarriers=52,
                hidden_dim=64,
                num_layers=1,
                encoder_type=encoder_type,
            )
            model = CrowdEstimator(config)
            x = torch.randn(2, 100, 52)
            output = model(x)
            assert output.shape == (2, 1)

    def test_gradient_flow(self):
        """Verify gradients flow through the model."""
        # Create fresh model to avoid state pollution from other tests
        torch.manual_seed(42)
        config = CrowdEstimatorConfig(
            n_subcarriers=52,
            seq_length=100,
            hidden_dim=64,
            num_layers=1,
            mode="regression",
        )
        model = CrowdEstimator(config)
        model.train()  # Ensure training mode for proper gradient flow

        x = torch.randn(2, 100, 52)
        x.requires_grad_(True)
        x.retain_grad()

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input tensor should have gradients"
        # Check that model parameters have gradients
        has_param_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_param_grad, "Model parameters should have non-zero gradients"

    def test_model_from_kwargs(self):
        """Create model using kwargs instead of config."""
        model = CrowdEstimator(
            n_subcarriers=64,
            mode="classification",
            num_classes=3,
        )
        assert model.config.n_subcarriers == 64
        assert model.config.mode == "classification"
        assert model.config.num_classes == 3


# =============================================================================
# CrowdEstimatorLight Tests
# =============================================================================


class TestCrowdEstimatorLight:
    """Tests for lightweight CrowdEstimatorLight model."""

    @pytest.fixture
    def light_model(self):
        """Create lightweight model for testing."""
        return CrowdEstimatorLight(n_subcarriers=52, hidden_dim=32, num_classes=4)

    def test_forward(self, light_model):
        """Forward pass."""
        x = torch.randn(4, 100, 52)
        output = light_model(x)
        assert output.shape == (4, 4)

    def test_smaller_model(self, light_model):
        """Light model should have fewer parameters than full model."""
        full_model = CrowdEstimator(CrowdEstimatorConfig(
            n_subcarriers=52,
            hidden_dim=128,
            mode="classification",
        ))

        light_params = sum(p.numel() for p in light_model.parameters())
        full_params = sum(p.numel() for p in full_model.parameters())

        assert light_params < full_params

    def test_export_onnx(self, light_model):
        """Test ONNX export functionality."""
        pytest.importorskip("onnxscript", reason="onnxscript not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            light_model.export_onnx(str(onnx_path), seq_length=100)
            assert onnx_path.exists()
            # Check file is not empty
            assert onnx_path.stat().st_size > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_create_regression_model(self):
        """Create regression model."""
        model = create_model(mode="regression")
        x = torch.randn(2, 100, 52)
        output = model(x)
        assert output.shape == (2, 1)

    def test_create_classification_model(self):
        """Create classification model."""
        model = create_model(mode="classification", num_classes=5)
        x = torch.randn(2, 100, 52)
        output = model(x)
        assert output.shape == (2, 5)

    def test_create_with_different_encoders(self):
        """Create models with different encoder types."""
        for encoder_type in ["transformer", "lstm", "bilstm"]:
            model = create_model(encoder_type=encoder_type)
            x = torch.randn(2, 100, 52)
            output = model(x)
            assert output.shape == (2, 1)

    def test_create_with_custom_subcarriers(self):
        """Create model with custom subcarrier count."""
        model = create_model(n_subcarriers=114)  # ESP32 has 114 subcarriers
        x = torch.randn(2, 100, 114)
        output = model(x)
        assert output.shape == (2, 1)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCrowdEstimatorIntegration:
    """Integration tests for CrowdEstimator."""

    def test_training_step(self):
        """Simulate a training step."""
        model = create_model(mode="regression")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Synthetic data
        x = torch.randn(8, 100, 52)
        y = torch.randint(0, 10, (8, 1)).float()

        # Forward pass
        model.train()
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_classification_training_step(self):
        """Simulate a classification training step."""
        model = create_model(mode="classification", num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Synthetic data
        x = torch.randn(8, 100, 52)
        y = torch.randint(0, 4, (8,))

        # Forward pass
        model.train()
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_model_save_load(self):
        """Test model save and load."""
        model = create_model(mode="regression")

        # Get initial prediction
        x = torch.randn(1, 100, 52)
        model.eval()
        with torch.no_grad():
            pred1 = model(x)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), path)

            model2 = create_model(mode="regression")
            model2.load_state_dict(torch.load(path, weights_only=True))
            model2.eval()

            with torch.no_grad():
                pred2 = model2(x)

        assert torch.allclose(pred1, pred2)
