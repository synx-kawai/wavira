"""
3D CNN-based Gesture Recognition Model

Implements gesture recognition from Wi-Fi CSI data using 3D Convolutional Neural Networks.
Based on the paper: "3次元CNNを利用したWi-Fi CSIによるジェスチャ認識" (Miyashiro & Miyashita)

The model uses temporal information by treating CSI sequences as 3D data:
- Dimension 1: Time (frames)
- Dimension 2: Routes (TX * RX antenna combinations)
- Dimension 3: Subcarriers

Architecture follows C3D (Tran et al., 2015) with modifications for CSI data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class GestureRecognizerConfig:
    """Configuration for gesture recognition model."""
    n_gestures: int = 8  # Number of gesture classes
    n_subcarriers: int = 114  # Number of subcarriers (ESP32: 114, Intel 5300: 30)
    n_routes: int = 3  # Number of TX*RX antenna routes
    n_frames: int = 32  # Number of frames per sample
    dropout_rate: float = 0.5  # Dropout rate for regularization
    upsample_size: Tuple[int, int] = (120, 120)  # Size for upsampling routes x subcarriers


class GestureRecognizer3DCNN(nn.Module):
    """
    3D CNN model for gesture recognition from CSI data.

    Processes CSI data as 3D volumes (frames × routes × subcarriers) using
    3D convolutions to capture spatiotemporal features.

    Args:
        config: Model configuration

    Input shape: (batch, 1, n_frames, n_routes, n_subcarriers)
    Output shape: (batch, n_gestures)
    """

    def __init__(self, config: Optional[GestureRecognizerConfig] = None):
        super().__init__()

        if config is None:
            config = GestureRecognizerConfig()

        self.config = config
        self.n_gestures = config.n_gestures
        self.upsample_size = config.upsample_size

        # 3D Convolutional layers following C3D architecture
        # All conv layers use 3x3x3 kernels with padding to preserve spatial dims

        # Block 1: 64 filters
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Block 2: 128 filters
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Block 3: 256 filters
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Block 4: 512 filters
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Block 5: 512 filters
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        # Calculate flattened size after convolutions
        # After upsampling to (32, 120, 120) and pooling:
        # pool1: (32, 60, 60)
        # pool2: (16, 30, 30)
        # pool3: (8, 15, 15)
        # pool4: (4, 7, 7)
        # pool5: (2, 4, 4)
        self._fc_input_size = 512 * 2 * 4 * 4  # 16384

        # Fully connected layers
        self.fc1 = nn.Linear(self._fc_input_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, config.n_gestures)

        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _upsample_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input CSI data to fixed size for consistent processing.

        The original paper upsamples to (n_frames, 120, 120) by repeating data
        to provide enough spatial information for convolutions.

        Args:
            x: Input tensor of shape (batch, 1, n_frames, n_routes, n_subcarriers)

        Returns:
            Upsampled tensor of shape (batch, 1, n_frames, 120, 120)
        """
        batch_size = x.shape[0]
        n_frames = x.shape[2]

        # Ensure tensor is contiguous before reshape
        x = x.contiguous()

        # Reshape to (batch * n_frames, 1, n_routes, n_subcarriers) for 2D interpolation
        x = x.reshape(batch_size * n_frames, 1, x.shape[3], x.shape[4])

        # Bilinear interpolation to target size
        x = F.interpolate(x, size=self.upsample_size, mode='bilinear', align_corners=False)

        # Reshape back to (batch, 1, n_frames, H, W)
        x = x.reshape(batch_size, 1, n_frames, self.upsample_size[0], self.upsample_size[1])

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input CSI data of shape (batch, n_routes, n_subcarriers, n_frames)
               or (batch, 1, n_frames, n_routes, n_subcarriers)

        Returns:
            Gesture class logits of shape (batch, n_gestures)
        """
        # Handle different input formats
        if x.dim() == 4:
            # Input: (batch, n_routes, n_subcarriers, n_frames)
            # -> (batch, 1, n_frames, n_routes, n_subcarriers)
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        elif x.dim() == 5 and x.shape[1] != 1:
            # Input: (batch, n_frames, n_routes, n_subcarriers, 1) or similar
            # Ensure channel is dim 1
            if x.shape[-1] == 1:
                x = x.squeeze(-1).unsqueeze(1)

        # Upsample to consistent size
        x = self._upsample_input(x)

        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(F.relu(self.bn3b(self.conv3b(x))))

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = self.pool4(F.relu(self.bn4b(self.conv4b(x))))

        x = F.relu(self.bn5a(self.conv5a(x)))
        x = self.pool5(F.relu(self.bn5b(self.conv5b(x))))

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.

        Args:
            x: Input CSI data

        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, predictions = torch.max(probs, dim=-1)
        return predictions, confidence


class GestureRecognizerLite(nn.Module):
    """
    Lightweight 3D CNN for gesture recognition.

    A smaller version suitable for edge deployment with fewer parameters
    while maintaining reasonable accuracy.

    Args:
        config: Model configuration
    """

    def __init__(self, config: Optional[GestureRecognizerConfig] = None):
        super().__init__()

        if config is None:
            config = GestureRecognizerConfig()

        self.config = config
        self.n_gestures = config.n_gestures

        # Smaller network without upsampling
        # Input: (batch, 1, n_frames, n_routes, n_subcarriers)

        # Block 1
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        # Block 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        # Block 3
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        # Block 4
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, config.n_gestures)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Handle different input formats
        if x.dim() == 4:
            # (batch, n_routes, n_subcarriers, n_frames) -> (batch, 1, n_frames, n_routes, n_subcarriers)
            x = x.permute(0, 3, 1, 2).unsqueeze(1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with confidence scores."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, predictions = torch.max(probs, dim=-1)
        return predictions, confidence


class DualESP32GestureRecognizer(nn.Module):
    """
    Gesture recognizer designed for dual ESP32 setup.

    This model processes CSI data from two ESP32 devices:
    - One ESP32 as transmitter (TX)
    - One ESP32 as receiver (RX)

    The architecture fuses information from both devices for improved recognition.

    Args:
        config: Model configuration
        fusion_method: How to fuse dual-device data ('concat', 'attention', 'add')
    """

    def __init__(
        self,
        config: Optional[GestureRecognizerConfig] = None,
        fusion_method: str = 'concat'
    ):
        super().__init__()

        if config is None:
            config = GestureRecognizerConfig()

        self.config = config
        self.fusion_method = fusion_method

        # Feature extractor (shared or separate for each ESP32)
        self.feature_extractor = self._build_feature_extractor()

        # Fusion and classifier
        # Feature extractor outputs 256-dim, so concat doubles to 512
        if fusion_method == 'concat':
            self.fc1 = nn.Linear(256 * 2, 256)  # 256 + 256 for concatenation
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
            self.fc1 = nn.Linear(256, 256)
        else:  # add
            self.fc1 = nn.Linear(256, 256)

        self.fc2 = nn.Linear(256, config.n_gestures)
        self.dropout = nn.Dropout(config.dropout_rate)

    def _build_feature_extractor(self) -> nn.Sequential:
        """Build the convolutional feature extractor."""
        return nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2)),

            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2)),

            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2)),

            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

    def forward(
        self,
        csi_device1: torch.Tensor,
        csi_device2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with dual-device CSI.

        Args:
            csi_device1: CSI from first ESP32 (batch, n_routes, n_subcarriers, n_frames)
            csi_device2: CSI from second ESP32 (optional, for single-device mode)

        Returns:
            Gesture class logits
        """
        # Reshape inputs if needed
        if csi_device1.dim() == 4:
            csi_device1 = csi_device1.permute(0, 3, 1, 2).unsqueeze(1)

        # Extract features from first device
        feat1 = self.feature_extractor(csi_device1)

        if csi_device2 is not None:
            if csi_device2.dim() == 4:
                csi_device2 = csi_device2.permute(0, 3, 1, 2).unsqueeze(1)
            feat2 = self.feature_extractor(csi_device2)

            # Fuse features
            if self.fusion_method == 'concat':
                x = torch.cat([feat1, feat2], dim=-1)
            elif self.fusion_method == 'attention':
                # Stack and apply attention
                feats = torch.stack([feat1, feat2], dim=1)
                x, _ = self.attention(feats, feats, feats)
                x = x.mean(dim=1)
            else:  # add
                x = feat1 + feat2
        else:
            # Single device mode - duplicate features for compatibility
            if self.fusion_method == 'concat':
                x = torch.cat([feat1, feat1], dim=-1)
            else:
                x = feat1

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# Default gesture labels following the paper
DEFAULT_GESTURE_LABELS = [
    "zoom_out",      # 1) Zoom Out - fingers spread apart
    "zoom_in",       # 2) Zoom In - fingers come together
    "circle_left",   # 3) Circle Left - finger circles counterclockwise
    "circle_right",  # 4) Circle Right - finger circles clockwise
    "swipe_left",    # 5) Swipe Left - hand swipes left
    "swipe_right",   # 6) Swipe Right - hand swipes right
    "flip_up",       # 7) Flip Up - hand flips upward
    "flip_down",     # 8) Flip Down - hand flips downward
]


def create_gesture_model(
    model_type: str = "standard",
    n_gestures: int = 8,
    n_subcarriers: int = 114,
    n_routes: int = 3,
    n_frames: int = 32,
    **kwargs
) -> nn.Module:
    """
    Factory function to create gesture recognition models.

    Args:
        model_type: Type of model ('standard', 'lite', 'dual')
        n_gestures: Number of gesture classes
        n_subcarriers: Number of CSI subcarriers
        n_routes: Number of TX*RX routes
        n_frames: Number of frames per sample
        **kwargs: Additional model-specific arguments

    Returns:
        Gesture recognition model
    """
    config = GestureRecognizerConfig(
        n_gestures=n_gestures,
        n_subcarriers=n_subcarriers,
        n_routes=n_routes,
        n_frames=n_frames,
    )

    if model_type == "standard":
        return GestureRecognizer3DCNN(config)
    elif model_type == "lite":
        return GestureRecognizerLite(config)
    elif model_type == "dual":
        fusion_method = kwargs.get("fusion_method", "concat")
        return DualESP32GestureRecognizer(config, fusion_method=fusion_method)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
