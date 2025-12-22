"""
Unit tests for Wavira models and components.
"""

import numpy as np
import pytest
import torch

from wavira.models.whofi import WhoFi, WhoFiWithClassifier
from wavira.models.encoder import TransformerEncoder, LSTMEncoder, BiLSTMEncoder, get_encoder
from wavira.data.preprocessing import (
    hampel_filter,
    phase_sanitization,
    extract_amplitude,
    preprocess_csi,
)
from wavira.data.dataset import CSIDataset, create_synthetic_csi_data
from wavira.losses.inbatch_loss import InBatchNegativeLoss, TripletLoss
from wavira.utils.metrics import compute_cmc, compute_map, evaluate_reid


class TestPreprocessing:
    """Tests for CSI preprocessing functions."""

    def test_extract_amplitude(self):
        """Test amplitude extraction from complex values."""
        # Create complex CSI data
        real = np.random.randn(3, 114, 100)
        imag = np.random.randn(3, 114, 100)
        csi_complex = real + 1j * imag

        amplitude = extract_amplitude(csi_complex)

        assert amplitude.shape == csi_complex.shape
        assert np.all(amplitude >= 0)
        assert np.allclose(amplitude, np.sqrt(real**2 + imag**2))

    def test_hampel_filter(self):
        """Test Hampel filter removes outliers."""
        # Create signal with outliers
        signal = np.zeros(100)
        signal[50] = 10  # Outlier

        filtered = hampel_filter(signal, window_size=5, threshold=3)

        # Outlier should be reduced
        assert abs(filtered[50]) < abs(signal[50])

    def test_phase_sanitization(self):
        """Test phase sanitization removes linear component."""
        n_subcarriers = 114
        n_packets = 100

        # Create phase with linear component
        subcarriers = np.arange(n_subcarriers)
        phase = np.outer(0.1 * subcarriers, np.ones(n_packets))  # Linear in subcarrier

        sanitized = phase_sanitization(phase)

        # Linear component should be removed
        # Check that variance across subcarriers is reduced
        assert np.var(sanitized, axis=0).mean() < np.var(phase, axis=0).mean()

    def test_preprocess_csi(self):
        """Test full preprocessing pipeline."""
        # Create complex CSI data
        csi = np.random.randn(3, 114, 200) + 1j * np.random.randn(3, 114, 200)

        processed = preprocess_csi(csi, use_phase=False)

        assert processed.shape == (3, 114, 200)
        assert processed.dtype == np.float32


class TestEncoders:
    """Tests for encoder architectures."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input for encoders."""
        batch_size = 4
        seq_len = 100
        input_dim = 342  # 3 * 114
        return torch.randn(batch_size, seq_len, input_dim)

    def test_transformer_encoder(self, sample_input):
        """Test Transformer encoder."""
        encoder = TransformerEncoder(
            input_dim=342,
            d_model=256,
            nhead=8,
            num_layers=1,
        )

        output = encoder(sample_input)

        assert output.shape == (4, 100, 256)

    def test_lstm_encoder(self, sample_input):
        """Test LSTM encoder."""
        encoder = LSTMEncoder(
            input_dim=342,
            hidden_dim=256,
            num_layers=1,
        )

        output = encoder(sample_input)

        assert output.shape == (4, 100, 256)

    def test_bilstm_encoder(self, sample_input):
        """Test Bi-LSTM encoder."""
        encoder = BiLSTMEncoder(
            input_dim=342,
            hidden_dim=128,
            num_layers=1,
        )

        output = encoder(sample_input)

        assert output.shape == (4, 100, 256)  # 128 * 2 for bidirectional

    def test_get_encoder(self):
        """Test encoder factory function."""
        for encoder_type in ["transformer", "lstm", "bilstm"]:
            encoder = get_encoder(encoder_type, input_dim=342, hidden_dim=256)
            assert encoder is not None


class TestWhoFiModel:
    """Tests for WhoFi model."""

    @pytest.fixture
    def sample_csi(self):
        """Create sample CSI input."""
        batch_size = 4
        n_channels = 3
        n_subcarriers = 114
        seq_len = 200
        return torch.randn(batch_size, n_channels, n_subcarriers, seq_len)

    def test_whofi_forward(self, sample_csi):
        """Test WhoFi forward pass."""
        model = WhoFi(
            n_channels=3,
            n_subcarriers=114,
            encoder_type="transformer",
            hidden_dim=256,
            signature_dim=256,
        )

        signature = model(sample_csi)

        assert signature.shape == (4, 256)
        # Check L2 normalization
        norms = torch.norm(signature, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_whofi_similarity(self, sample_csi):
        """Test similarity computation."""
        model = WhoFi()

        sig1 = model(sample_csi)
        sig2 = model(sample_csi)

        similarity = model.compute_similarity(sig1, sig2)

        assert similarity.shape == (4, 4)
        # Self-similarity should be high
        assert torch.allclose(torch.diag(similarity), torch.ones(4), atol=1e-5)

    def test_whofi_encoder_types(self, sample_csi):
        """Test WhoFi with different encoder types."""
        for encoder_type in ["transformer", "lstm", "bilstm"]:
            model = WhoFi(encoder_type=encoder_type)
            signature = model(sample_csi)
            assert signature.shape == (4, 256)

    def test_whofi_with_classifier(self, sample_csi):
        """Test WhoFi with classification head."""
        model = WhoFiWithClassifier(n_classes=14)

        logits, signature = model(sample_csi)

        assert logits.shape == (4, 14)
        assert signature.shape == (4, 256)


class TestDataset:
    """Tests for dataset classes."""

    def test_create_synthetic_data(self):
        """Test synthetic data generation."""
        data, labels = create_synthetic_csi_data(
            n_persons=10,
            samples_per_person=5,
            n_rx_antennas=3,
            n_subcarriers=114,
            n_packets=100,
        )

        assert data.shape == (50, 3, 114, 100)
        assert labels.shape == (50,)
        assert len(np.unique(labels)) == 10

    def test_csi_dataset(self):
        """Test CSI dataset."""
        data, labels = create_synthetic_csi_data(
            n_persons=5,
            samples_per_person=10,
        )

        dataset = CSIDataset(
            data=data,
            labels=labels,
            sequence_length=200,
            preprocess=False,
            normalize=True,
        )

        assert len(dataset) == 50
        assert dataset.n_classes == 5

        sample, label = dataset[0]
        assert sample.shape[2] == 200  # sequence length


class TestLossFunctions:
    """Tests for loss functions."""

    @pytest.fixture
    def sample_signatures(self):
        """Create sample signatures."""
        signatures = torch.randn(8, 256)
        signatures = torch.nn.functional.normalize(signatures, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        return signatures, labels

    def test_inbatch_negative_loss(self, sample_signatures):
        """Test in-batch negative loss."""
        signatures, labels = sample_signatures

        loss_fn = InBatchNegativeLoss(temperature=0.07)
        loss = loss_fn(signatures, labels)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_triplet_loss(self):
        """Test triplet loss."""
        anchor = torch.randn(4, 256)
        positive = anchor + 0.1 * torch.randn(4, 256)
        negative = torch.randn(4, 256)

        loss_fn = TripletLoss(margin=0.3)
        loss = loss_fn(anchor, positive, negative)

        assert loss.shape == ()
        assert loss.item() >= 0


class TestMetrics:
    """Tests for evaluation metrics."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features with known structure."""
        # Create features where same-class features are similar
        n_classes = 5
        samples_per_class = 10

        features = []
        labels = []

        for c in range(n_classes):
            class_center = np.random.randn(256)
            for _ in range(samples_per_class):
                sample = class_center + 0.1 * np.random.randn(256)
                sample = sample / np.linalg.norm(sample)
                features.append(sample)
                labels.append(c)

        return np.array(features), np.array(labels)

    def test_compute_cmc(self, sample_features):
        """Test CMC curve computation."""
        features, labels = sample_features

        from wavira.utils.metrics import compute_distance_matrix
        dist_matrix = compute_distance_matrix(features, features, metric="cosine")
        cmc = compute_cmc(dist_matrix, labels, labels, max_rank=10)

        assert cmc.shape == (10,)
        # CMC should be monotonically increasing
        assert np.all(np.diff(cmc) >= 0)
        # With well-separated classes, Rank-1 should be high
        assert cmc[0] > 0.5

    def test_compute_map(self, sample_features):
        """Test mAP computation."""
        features, labels = sample_features

        from wavira.utils.metrics import compute_distance_matrix
        dist_matrix = compute_distance_matrix(features, features, metric="cosine")
        mAP = compute_map(dist_matrix, labels, labels)

        assert 0 <= mAP <= 1
        # With well-separated classes, mAP should be reasonable
        assert mAP > 0.3

    def test_evaluate_reid(self, sample_features):
        """Test full evaluation."""
        features, labels = sample_features

        metrics = evaluate_reid(
            features, features,
            labels, labels,
            metric="cosine",
            ranks=(1, 5, 10),
        )

        assert "mAP" in metrics
        assert "Rank-1" in metrics
        assert "Rank-5" in metrics
        assert "Rank-10" in metrics

        assert metrics["Rank-1"] <= metrics["Rank-5"] <= metrics["Rank-10"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
