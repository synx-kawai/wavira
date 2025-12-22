"""
WhoFi: Wi-Fi-based Person Re-Identification Model

Main model architecture combining:
- CSI input processing
- Encoder (Transformer/LSTM/Bi-LSTM)
- Signature module for biometric feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from wavira.models.encoder import get_encoder


class SignatureModule(nn.Module):
    """
    Signature module for generating biometric signatures.

    Takes encoder output and produces L2-normalized signature vectors
    for person re-identification.
    """

    def __init__(
        self,
        input_dim: int,
        signature_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize signature module.

        Args:
            input_dim: Input dimension from encoder
            signature_dim: Output signature vector dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, signature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(signature_dim, signature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate L2-normalized signature vector.

        Args:
            x: Encoder output of shape (batch, hidden_dim)

        Returns:
            Normalized signature of shape (batch, signature_dim)
        """
        signature = self.projection(x)
        # L2 normalization
        signature = F.normalize(signature, p=2, dim=-1)
        return signature


class WhoFi(nn.Module):
    """
    WhoFi: Deep Person Re-Identification via Wi-Fi Channel Signal Encoding

    Complete architecture for extracting biometric signatures from CSI data.

    Architecture:
    1. Input: CSI data (channels x subcarriers x packets)
    2. Flatten channels and subcarriers -> (batch, seq_len, features)
    3. Encoder: Transformer/LSTM/Bi-LSTM
    4. Pooling: Mean pooling over sequence
    5. Signature: Linear + L2 normalization

    The model outputs normalized signature vectors that can be compared
    using cosine similarity for person re-identification.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_subcarriers: int = 114,
        encoder_type: str = "transformer",
        hidden_dim: int = 256,
        signature_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        nhead: int = 8,
        pooling: str = "mean",
    ):
        """
        Initialize WhoFi model.

        Args:
            n_channels: Number of input channels (e.g., antennas)
            n_subcarriers: Number of subcarriers
            encoder_type: Type of encoder ('transformer', 'lstm', 'bilstm')
            hidden_dim: Hidden dimension for encoder
            signature_dim: Output signature dimension
            num_layers: Number of encoder layers
            dropout: Dropout probability
            nhead: Number of attention heads (for transformer)
            pooling: Pooling strategy ('mean', 'max', 'cls')
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_subcarriers = n_subcarriers
        self.input_dim = n_channels * n_subcarriers
        self.hidden_dim = hidden_dim
        self.signature_dim = signature_dim
        self.pooling = pooling

        # Encoder
        self.encoder = get_encoder(
            encoder_type=encoder_type,
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
        )

        # Determine encoder output dimension
        if encoder_type.lower() == "bilstm":
            encoder_output_dim = hidden_dim
        else:
            encoder_output_dim = hidden_dim

        # CLS token for classification pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim))

        # Signature module
        self.signature = SignatureModule(
            input_dim=encoder_output_dim,
            signature_dim=signature_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input CSI tensor of shape (batch, channels, subcarriers, seq_len)
            return_features: If True, also return intermediate features

        Returns:
            Signature vector of shape (batch, signature_dim)
            If return_features=True, returns (signature, encoder_output)

        Raises:
            ValueError: If input tensor has incorrect shape
        """
        # Input shape validation
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (batch, channels, subcarriers, seq_len), "
                f"got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )

        if x.size(1) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {x.size(1)}. "
                f"Input shape: {tuple(x.shape)}, expected: (batch, {self.n_channels}, {self.n_subcarriers}, seq_len)"
            )

        if x.size(2) != self.n_subcarriers:
            raise ValueError(
                f"Expected {self.n_subcarriers} subcarriers, got {x.size(2)}. "
                f"Input shape: {tuple(x.shape)}, expected: (batch, {self.n_channels}, {self.n_subcarriers}, seq_len)"
            )

        batch_size = x.size(0)

        # Reshape: (batch, channels, subcarriers, seq) -> (batch, seq, channels * subcarriers)
        x = x.permute(0, 3, 1, 2)  # (batch, seq, channels, subcarriers)
        x = x.reshape(batch_size, -1, self.input_dim)  # (batch, seq, input_dim)

        # Add CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Encode
        encoded = self.encoder(x)

        # Pool over sequence dimension
        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        elif self.pooling == "max":
            pooled = encoded.max(dim=1)[0]
        elif self.pooling == "cls":
            pooled = encoded[:, 0]  # Use CLS token output
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Generate signature
        signature = self.signature(pooled)

        if return_features:
            return signature, encoded

        return signature

    def compute_similarity(
        self,
        query: torch.Tensor,
        gallery: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and gallery signatures.

        Args:
            query: Query signatures of shape (n_query, signature_dim)
            gallery: Gallery signatures of shape (n_gallery, signature_dim)

        Returns:
            Similarity matrix of shape (n_query, n_gallery)
        """
        # Both should already be L2 normalized
        similarity = torch.mm(query, gallery.t())
        return similarity

    def extract_signatures(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract signatures for all samples in a dataloader.

        Args:
            dataloader: DataLoader containing CSI samples
            device: Device to run inference on

        Returns:
            Tuple of (signatures, labels)
        """
        self.eval()
        all_signatures = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, labels = batch
                else:
                    x = batch[0]
                    labels = batch[-1]

                x = x.to(device)
                signatures = self(x)

                all_signatures.append(signatures.cpu())
                all_labels.append(labels)

        signatures = torch.cat(all_signatures, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return signatures, labels


class WhoFiWithClassifier(nn.Module):
    """
    WhoFi model with additional classification head.

    Useful for joint training with cross-entropy loss and signature loss.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 3,
        n_subcarriers: int = 114,
        encoder_type: str = "transformer",
        hidden_dim: int = 256,
        signature_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        nhead: int = 8,
    ):
        """
        Initialize WhoFi with classifier.

        Args:
            n_classes: Number of identity classes
            Other args: Same as WhoFi
        """
        super().__init__()

        self.whofi = WhoFi(
            n_channels=n_channels,
            n_subcarriers=n_subcarriers,
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            signature_dim=signature_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
        )

        self.classifier = nn.Linear(signature_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_signature: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input CSI tensor
            return_signature: Whether to return signature along with logits

        Returns:
            Tuple of (class_logits, signature) if return_signature=True
            Otherwise just class_logits
        """
        signature = self.whofi(x)
        logits = self.classifier(signature)

        if return_signature:
            return logits, signature
        return logits
