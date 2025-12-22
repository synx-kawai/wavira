"""
Encoder Architectures for CSI Feature Extraction

Implements three encoder types as described in the WhoFi paper:
- Transformer: Best performance with attention mechanism
- LSTM: Recurrent network for sequential modeling
- Bi-LSTM: Bidirectional LSTM for capturing both directions
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds position information to the input embeddings using
    sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Position-encoded tensor
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for CSI sequences.

    Uses multi-head self-attention to model long-range temporal
    dependencies in the CSI signal. According to the paper,
    a single encoder layer achieves optimal performance.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 2000,
    ):
        """
        Initialize Transformer encoder.

        Args:
            input_dim: Input feature dimension (n_channels * n_subcarriers)
            d_model: Model dimension for attention
            nhead: Number of attention heads
            num_layers: Number of encoder layers (paper recommends 1)
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Encoded features of shape (batch, seq_len, d_model)
        """
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        output = self.transformer(x, mask=mask)

        return output


class LSTMEncoder(nn.Module):
    """
    LSTM-based encoder for CSI sequences.

    Standard unidirectional LSTM for sequential modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize LSTM encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional initial hidden state

        Returns:
            Encoded features of shape (batch, seq_len, hidden_dim)
        """
        output, _ = self.lstm(x, hidden)
        return self.dropout(output)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for CSI sequences.

    Captures temporal patterns in both forward and backward directions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize Bi-LSTM encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension (per direction)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 2  # Bidirectional doubles output

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional initial hidden state

        Returns:
            Encoded features of shape (batch, seq_len, hidden_dim * 2)
        """
        output, _ = self.lstm(x, hidden)
        return self.dropout(output)


def get_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 1,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create encoder by type.

    Args:
        encoder_type: One of 'transformer', 'lstm', 'bilstm'
        input_dim: Input feature dimension
        hidden_dim: Hidden/model dimension
        num_layers: Number of layers
        dropout: Dropout probability
        **kwargs: Additional encoder-specific arguments

    Returns:
        Encoder module
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "transformer":
        return TransformerEncoder(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=kwargs.get("nhead", 8),
            dim_feedforward=kwargs.get("dim_feedforward", hidden_dim * 2),
            max_seq_len=kwargs.get("max_seq_len", 2000),
        )
    elif encoder_type == "lstm":
        return LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "bilstm":
        return BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,  # Half for each direction
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
