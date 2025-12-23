"""
Crowd Level Estimation Model using WiFi CSI

Issue #5: 混雑レベル推定モデルの開発

Supports two modes:
- Regression: Predict exact number of people
- Classification: Predict crowd level category

Architecture based on WhoFi encoder with modified head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from wavira.models.encoder import get_encoder


@dataclass
class CrowdEstimatorConfig:
    """Configuration for CrowdEstimator model."""
    n_subcarriers: int = 52
    seq_length: int = 100  # Number of time steps (10 seconds at 10Hz)
    encoder_type: str = "transformer"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    nhead: int = 4
    max_people: int = 10  # Maximum number of people to estimate
    num_classes: int = 4  # For classification: empty, low, medium, high
    mode: str = "regression"  # "regression" or "classification"


class TemporalBlock(nn.Module):
    """1D Convolutional block for temporal feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (batch, channels, seq_len)"""
        residual = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual


class CrowdEstimator(nn.Module):
    """
    WiFi CSI-based Crowd Level Estimator.

    Takes CSI amplitude data and estimates the number of people
    or classifies crowd density level.

    Input: (batch, seq_len, n_subcarriers) - CSI amplitude time series
    Output:
        - Regression: (batch, 1) - estimated person count
        - Classification: (batch, num_classes) - class logits
    """

    def __init__(self, config: Optional[CrowdEstimatorConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = CrowdEstimatorConfig(**kwargs)
        self.config = config

        # Feature extraction with 1D CNN
        self.temporal_conv = nn.Sequential(
            TemporalBlock(config.n_subcarriers, 64, kernel_size=5, dropout=config.dropout),
            TemporalBlock(64, 128, kernel_size=3, dilation=2, dropout=config.dropout),
            TemporalBlock(128, config.hidden_dim, kernel_size=3, dilation=4, dropout=config.dropout),
        )

        # Encoder (Transformer/LSTM/BiLSTM)
        self.encoder = get_encoder(
            encoder_type=config.encoder_type,
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            nhead=config.nhead,
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Output heads
        if config.mode == "regression":
            self.head = nn.Sequential(
                nn.Linear(config.hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(64, 1),
                nn.ReLU(),  # People count is non-negative
            )
        else:  # classification
            self.head = nn.Sequential(
                nn.Linear(config.hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(64, config.num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input CSI tensor of shape (batch, seq_len, n_subcarriers)
            return_features: If True, also return intermediate features

        Returns:
            - Regression: (batch, 1) estimated person count
            - Classification: (batch, num_classes) class logits
        """
        batch_size = x.size(0)

        # x: (batch, seq_len, n_subcarriers) -> (batch, n_subcarriers, seq_len)
        x = x.permute(0, 2, 1)

        # Temporal convolution for local feature extraction
        x = self.temporal_conv(x)  # (batch, hidden_dim, seq_len)

        # Prepare for encoder: (batch, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)

        # Encode temporal dependencies
        encoded = self.encoder(x)  # (batch, seq_len, hidden_dim)

        # Attention pooling
        attn_weights = self.attention(encoded)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(encoded * attn_weights, dim=1)  # (batch, hidden_dim)

        # Output
        output = self.head(pooled)

        if return_features:
            return output, pooled
        return output

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Make prediction with additional metadata.

        Args:
            x: Input CSI tensor

        Returns:
            Dictionary with prediction and metadata
        """
        self.eval()
        with torch.no_grad():
            output = self(x)

            if self.config.mode == "regression":
                count = output.squeeze(-1)
                return {
                    "count": count.cpu().numpy(),
                    "count_int": count.round().long().cpu().numpy(),
                }
            else:
                probs = F.softmax(output, dim=-1)
                pred_class = output.argmax(dim=-1)
                labels = ["empty", "low", "medium", "high"]
                return {
                    "class": pred_class.cpu().numpy(),
                    "label": [labels[i] for i in pred_class.cpu().numpy()],
                    "probabilities": probs.cpu().numpy(),
                }


class CrowdEstimatorLight(nn.Module):
    """
    Lightweight version for edge deployment (ESP32).

    Uses simpler architecture suitable for microcontroller inference.
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        hidden_dim: int = 32,
        num_classes: int = 4,
    ):
        super().__init__()

        self.n_subcarriers = n_subcarriers

        # Simple 1D CNN
        self.features = nn.Sequential(
            nn.Conv1d(n_subcarriers, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (batch, seq_len, n_subcarriers)"""
        x = x.permute(0, 2, 1)  # (batch, n_subcarriers, seq_len)
        x = self.features(x)    # (batch, hidden_dim, 1)
        x = x.squeeze(-1)       # (batch, hidden_dim)
        x = self.classifier(x)  # (batch, num_classes)
        return x

    def export_onnx(self, filepath: str, seq_length: int = 100):
        """Export model to ONNX format for edge deployment."""
        dummy_input = torch.randn(1, seq_length, self.n_subcarriers)
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            input_names=['csi_input'],
            output_names=['crowd_level'],
            dynamic_axes={
                'csi_input': {0: 'batch_size', 1: 'seq_length'},
                'crowd_level': {0: 'batch_size'}
            },
            opset_version=11,
        )
        print(f"Model exported to {filepath}")


def create_model(
    mode: str = "regression",
    encoder_type: str = "transformer",
    n_subcarriers: int = 52,
    **kwargs
) -> CrowdEstimator:
    """
    Factory function to create CrowdEstimator model.

    Args:
        mode: "regression" or "classification"
        encoder_type: "transformer", "lstm", or "bilstm"
        n_subcarriers: Number of CSI subcarriers
        **kwargs: Additional config parameters

    Returns:
        CrowdEstimator model
    """
    config = CrowdEstimatorConfig(
        mode=mode,
        encoder_type=encoder_type,
        n_subcarriers=n_subcarriers,
        **kwargs
    )
    return CrowdEstimator(config)
