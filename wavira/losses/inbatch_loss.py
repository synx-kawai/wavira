"""
Loss Functions for Person Re-Identification

Implements:
- In-batch negative loss (as described in WhoFi paper)
- Triplet loss for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InBatchNegativeLoss(nn.Module):
    """
    In-batch negative loss for learning discriminative signatures.

    This loss treats all samples in a batch as potential matches and
    uses cross-entropy to maximize similarity for same-identity pairs
    (positives) and minimize it for different-identity pairs (negatives).

    The loss builds a similarity matrix S where S[i,j] = cosine_sim(sig_i, sig_j),
    then applies cross-entropy across each row to push the diagonal (positive)
    scores high and off-diagonal (negative) scores low.

    For batches with multiple samples per identity, we use the label information
    to identify all positive pairs, not just diagonal elements.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True,
    ):
        """
        Initialize in-batch negative loss.

        Args:
            temperature: Temperature scaling for similarity scores.
                         Lower values make the distribution sharper.
            symmetric: Whether to compute loss in both directions (q->g and g->q)
        """
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        signatures: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute in-batch negative loss.

        Args:
            signatures: L2-normalized signature vectors of shape (batch_size, dim)
            labels: Identity labels of shape (batch_size,)

        Returns:
            Scalar loss value
        """
        batch_size = signatures.size(0)
        device = signatures.device

        # Compute similarity matrix: S[i,j] = sig_i · sig_j^T
        # Since signatures are L2-normalized, this is cosine similarity
        similarity = torch.mm(signatures, signatures.t()) / self.temperature

        # Create positive mask: 1 where labels match
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()

        # Remove self-similarity from positives
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - eye_mask

        # For each anchor, we want to maximize similarity with positives
        # and minimize with negatives using cross-entropy

        # Mask out self-similarity with large negative value
        similarity = similarity - eye_mask * 1e9

        # Count positives per sample
        positive_counts = positive_mask.sum(dim=1)

        # Handle samples without positives (use self as positive via softmax target)
        has_positives = positive_counts > 0

        # Compute loss for samples with positives
        if has_positives.any():
            # Get log softmax over similarity scores
            log_prob = F.log_softmax(similarity, dim=1)

            # Compute mean log probability for positive pairs
            # Masked mean: sum(log_prob * mask) / sum(mask)
            positive_log_prob = (log_prob * positive_mask).sum(dim=1)
            positive_log_prob = positive_log_prob / positive_counts.clamp(min=1)

            # Average over samples that have positives
            loss = -positive_log_prob[has_positives].mean()
        else:
            # Fallback: self-supervised contrastive loss (InfoNCE-style)
            # When no positive pairs exist, use uniformity loss to spread embeddings
            # This encourages the model to produce diverse representations
            similarity_for_uniformity = torch.mm(signatures, signatures.t())
            # Exclude self-similarity
            similarity_for_uniformity = similarity_for_uniformity - eye_mask * 1e9
            # Uniformity loss: penalize high similarity between different samples
            loss = torch.logsumexp(similarity_for_uniformity / self.temperature, dim=1).mean()

        if self.symmetric:
            # Compute symmetric loss (transpose similarity matrix)
            similarity_t = similarity.t()
            log_prob_t = F.log_softmax(similarity_t, dim=1)
            positive_log_prob_t = (log_prob_t * positive_mask.t()).sum(dim=1)
            positive_log_prob_t = positive_log_prob_t / positive_counts.clamp(min=1)

            if has_positives.any():
                loss_t = -positive_log_prob_t[has_positives].mean()
                loss = (loss + loss_t) / 2

        return loss


class InBatchNegativeLossSimple(nn.Module):
    """
    Simplified in-batch negative loss assuming each sample is unique.

    In this version, we assume batch contains one sample per identity,
    and we compare query signatures with gallery signatures.
    This is closer to the original formulation in the paper.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        query_signatures: torch.Tensor,
        gallery_signatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute in-batch negative loss for query-gallery pairs.

        Assumes query[i] and gallery[i] are the same identity.

        Args:
            query_signatures: Query signatures of shape (batch_size, dim)
            gallery_signatures: Gallery signatures of shape (batch_size, dim)

        Returns:
            Scalar loss value
        """
        batch_size = query_signatures.size(0)
        device = query_signatures.device

        # Compute similarity matrix
        similarity = torch.mm(query_signatures, gallery_signatures.t()) / self.temperature

        # Target: diagonal should be maximum (matching pairs)
        targets = torch.arange(batch_size, device=device)

        # Cross-entropy loss: maximize diagonal, minimize off-diagonal
        loss = self.cross_entropy(similarity, targets)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.

    Pushes anchor closer to positive and further from negative.
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining: str = "hard",
    ):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
            mining: Mining strategy ('hard', 'semi-hard', 'all')
        """
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor signatures of shape (batch_size, dim)
            positive: Positive signatures of shape (batch_size, dim)
            negative: Negative signatures of shape (batch_size, dim)

        Returns:
            Scalar loss value
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        # Triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)

        return losses.mean()

    def forward_with_labels(
        self,
        signatures: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with online mining.

        Args:
            signatures: All signatures of shape (batch_size, dim)
            labels: Identity labels of shape (batch_size,)

        Returns:
            Scalar loss value
        """
        batch_size = signatures.size(0)
        device = signatures.device

        # Compute pairwise distances
        dist_matrix = torch.cdist(signatures, signatures, p=2)

        # Create mask for positive and negative pairs
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = 1 - positive_mask

        # Remove diagonal
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - eye_mask

        if self.mining == "hard":
            # Hard positive: furthest positive
            pos_dist = dist_matrix * positive_mask
            pos_dist[positive_mask == 0] = 0
            hardest_pos = pos_dist.max(dim=1)[0]

            # Hard negative: closest negative
            neg_dist = dist_matrix.clone()
            neg_dist[negative_mask == 0] = float('inf')
            hardest_neg = neg_dist.min(dim=1)[0]

            losses = F.relu(hardest_pos - hardest_neg + self.margin)
            # Only count valid triplets
            valid = (hardest_pos > 0) & (hardest_neg < float('inf'))
            if valid.sum() > 0:
                return losses[valid].mean()
            return torch.tensor(0.0, device=device)

        else:  # all triplets
            total_loss = 0
            count = 0

            for i in range(batch_size):
                pos_indices = (positive_mask[i] == 1).nonzero(as_tuple=True)[0]
                neg_indices = (negative_mask[i] == 1).nonzero(as_tuple=True)[0]

                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                for pos_idx in pos_indices:
                    for neg_idx in neg_indices:
                        pos_dist = dist_matrix[i, pos_idx]
                        neg_dist = dist_matrix[i, neg_idx]
                        loss = F.relu(pos_dist - neg_dist + self.margin)
                        total_loss += loss
                        count += 1

            if count > 0:
                return total_loss / count
            return torch.tensor(0.0, device=device)
