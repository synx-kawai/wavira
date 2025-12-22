"""
Unit tests for loss functions.
"""

import pytest
import torch

from wavira.losses.inbatch_loss import InBatchNegativeLoss, TripletLoss


class TestInBatchNegativeLoss:
    """Tests for InBatchNegativeLoss class."""

    def test_basic_forward(self):
        """Test basic forward pass with positive pairs."""
        loss_fn = InBatchNegativeLoss(temperature=0.07, symmetric=True)

        batch_size = 8
        dim = 64
        # Create signatures with some same labels
        signatures = torch.randn(batch_size, dim, requires_grad=True)
        signatures_norm = torch.nn.functional.normalize(signatures, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = loss_fn(signatures_norm, labels)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_no_positive_pairs_fallback(self):
        """Test fallback when no positive pairs exist."""
        loss_fn = InBatchNegativeLoss(temperature=0.07, symmetric=True)

        batch_size = 4
        dim = 64
        # Each sample has unique label - no positive pairs
        signatures = torch.randn(batch_size, dim, requires_grad=True)
        signatures_norm = torch.nn.functional.normalize(signatures, dim=1)
        labels = torch.tensor([0, 1, 2, 3])

        loss = loss_fn(signatures_norm, labels)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        loss_fn = InBatchNegativeLoss(temperature=0.07, symmetric=True)

        signatures = torch.randn(4, 32, requires_grad=True)
        signatures_norm = torch.nn.functional.normalize(signatures, dim=1)
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(signatures_norm, labels)
        loss.backward()

        assert signatures.grad is not None
        assert not torch.isnan(signatures.grad).any()

    def test_gradient_flow_no_positives(self):
        """Test gradient flow when no positive pairs exist."""
        loss_fn = InBatchNegativeLoss(temperature=0.07, symmetric=True)

        signatures = torch.randn(4, 32, requires_grad=True)
        signatures_norm = torch.nn.functional.normalize(signatures, dim=1)
        labels = torch.tensor([0, 1, 2, 3])  # All unique

        loss = loss_fn(signatures_norm, labels)
        loss.backward()

        assert signatures.grad is not None
        assert not torch.isnan(signatures.grad).any()

    def test_symmetric_loss(self):
        """Test symmetric loss computation."""
        loss_symmetric = InBatchNegativeLoss(temperature=0.07, symmetric=True)
        loss_asymmetric = InBatchNegativeLoss(temperature=0.07, symmetric=False)

        signatures = torch.randn(4, 32)
        signatures = torch.nn.functional.normalize(signatures, dim=1)
        labels = torch.tensor([0, 0, 1, 1])

        loss_s = loss_symmetric(signatures, labels)
        loss_a = loss_asymmetric(signatures, labels)

        # Both should be valid
        assert not torch.isnan(loss_s)
        assert not torch.isnan(loss_a)

    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        loss_low_temp = InBatchNegativeLoss(temperature=0.01, symmetric=False)
        loss_high_temp = InBatchNegativeLoss(temperature=1.0, symmetric=False)

        signatures = torch.randn(4, 32)
        signatures = torch.nn.functional.normalize(signatures, dim=1)
        labels = torch.tensor([0, 0, 1, 1])

        loss_lt = loss_low_temp(signatures, labels)
        loss_ht = loss_high_temp(signatures, labels)

        # Lower temperature should generally give higher loss
        # (but this depends on the random signatures)
        assert loss_lt != loss_ht


class TestTripletLoss:
    """Tests for TripletLoss class."""

    def test_basic_triplet_loss(self):
        """Test basic triplet loss computation."""
        loss_fn = TripletLoss(margin=0.3)

        batch_size = 8
        dim = 64
        anchor = torch.randn(batch_size, dim)
        positive = anchor + torch.randn(batch_size, dim) * 0.1  # Close to anchor
        negative = torch.randn(batch_size, dim)  # Random

        loss = loss_fn(anchor, positive, negative)

        assert loss.dim() == 0
        assert loss >= 0  # Triplet loss is non-negative

    def test_triplet_loss_with_labels(self):
        """Test triplet loss with online mining."""
        loss_fn = TripletLoss(margin=0.3, mining="hard")

        signatures = torch.randn(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = loss_fn.forward_with_labels(signatures, labels)

        assert loss.dim() == 0
        assert loss >= 0
