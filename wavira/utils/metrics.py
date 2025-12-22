"""
Evaluation Metrics for Person Re-Identification

Implements:
- Cumulative Matching Characteristic (CMC) curve
- Mean Average Precision (mAP)
- Rank-k accuracy
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional


def compute_distance_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute distance matrix between query and gallery features.

    Args:
        query_features: Query feature matrix of shape (n_query, dim)
        gallery_features: Gallery feature matrix of shape (n_gallery, dim)
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        Distance matrix of shape (n_query, n_gallery)
    """
    if metric == "cosine":
        # Normalize features
        query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity (higher is better)
        similarity = np.dot(query_norm, gallery_norm.T)

        # Convert to distance (lower is better)
        distance = 1 - similarity

    elif metric == "euclidean":
        # Euclidean distance
        query_sq = np.sum(query_features ** 2, axis=1, keepdims=True)
        gallery_sq = np.sum(gallery_features ** 2, axis=1, keepdims=True)
        cross = np.dot(query_features, gallery_features.T)

        distance = query_sq - 2 * cross + gallery_sq.T
        distance = np.sqrt(np.maximum(distance, 0))

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distance


def compute_cmc(
    distance_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 50,
) -> np.ndarray:
    """
    Compute Cumulative Matching Characteristic (CMC) curve.

    CMC@k is the probability that the correct match appears
    in the top-k retrieved results.

    Args:
        distance_matrix: Distance matrix of shape (n_query, n_gallery)
        query_labels: Query identity labels
        gallery_labels: Gallery identity labels
        max_rank: Maximum rank to compute

    Returns:
        CMC curve array of length max_rank
    """
    n_query = len(query_labels)
    max_rank = min(max_rank, len(gallery_labels))

    cmc = np.zeros(max_rank)

    for i in range(n_query):
        # Sort gallery by distance to query
        indices = np.argsort(distance_matrix[i])
        sorted_labels = gallery_labels[indices]

        # Find where correct labels appear
        matches = sorted_labels == query_labels[i]

        if matches.any():
            # First correct match position (0-indexed)
            first_match = np.where(matches)[0][0]

            # All ranks >= first_match are successful
            if first_match < max_rank:
                cmc[first_match:] += 1

    # Normalize to get probability
    cmc = cmc / n_query

    return cmc


def compute_average_precision(
    distances: np.ndarray,
    query_label: int,
    gallery_labels: np.ndarray,
) -> float:
    """
    Compute Average Precision for a single query.

    Args:
        distances: Distances from query to all gallery samples
        query_label: Query identity label
        gallery_labels: Gallery identity labels

    Returns:
        Average precision value
    """
    # Sort by distance
    indices = np.argsort(distances)
    sorted_labels = gallery_labels[indices]

    # Find positive matches
    matches = sorted_labels == query_label

    if not matches.any():
        return 0.0

    # Compute precision at each recall point
    n_positives = matches.sum()
    cumsum = np.cumsum(matches)
    precision_at_k = cumsum / (np.arange(len(matches)) + 1)

    # Average precision: mean of precision at each positive position
    ap = (precision_at_k * matches).sum() / n_positives

    return ap


def compute_map(
    distance_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """
    Compute mean Average Precision (mAP).

    Args:
        distance_matrix: Distance matrix of shape (n_query, n_gallery)
        query_labels: Query identity labels
        gallery_labels: Gallery identity labels

    Returns:
        mAP value
    """
    n_query = len(query_labels)
    aps = []

    for i in range(n_query):
        ap = compute_average_precision(
            distance_matrix[i],
            query_labels[i],
            gallery_labels,
        )
        aps.append(ap)

    return np.mean(aps)


def compute_rank_k_accuracy(
    distance_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int = 1,
) -> float:
    """
    Compute Rank-k accuracy.

    Args:
        distance_matrix: Distance matrix
        query_labels: Query identity labels
        gallery_labels: Gallery identity labels
        k: Rank threshold

    Returns:
        Rank-k accuracy (proportion of queries with correct match in top-k)
    """
    cmc = compute_cmc(distance_matrix, query_labels, gallery_labels, max_rank=k)
    return cmc[k - 1]


def evaluate_reid(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    metric: str = "cosine",
    ranks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Full evaluation for person re-identification.

    Args:
        query_features: Query feature matrix
        gallery_features: Gallery feature matrix
        query_labels: Query identity labels
        gallery_labels: Gallery identity labels
        metric: Distance metric
        ranks: Rank values to compute accuracy for

    Returns:
        Dictionary with mAP and Rank-k accuracies
    """
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(query_features, gallery_features, metric)

    # Compute mAP
    mAP = compute_map(distance_matrix, query_labels, gallery_labels)

    # Compute CMC curve
    max_rank = max(ranks)
    cmc = compute_cmc(distance_matrix, query_labels, gallery_labels, max_rank)

    # Build results dictionary
    results = {"mAP": mAP}
    for k in ranks:
        results[f"Rank-{k}"] = cmc[k - 1]

    return results


def evaluate_reid_torch(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    metric: str = "cosine",
    ranks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Evaluate re-identification performance with PyTorch tensors.

    Args:
        query_features: Query feature tensor
        gallery_features: Gallery feature tensor
        query_labels: Query label tensor
        gallery_labels: Gallery label tensor
        metric: Distance metric
        ranks: Rank values to compute

    Returns:
        Dictionary with evaluation metrics
    """
    return evaluate_reid(
        query_features.cpu().numpy(),
        gallery_features.cpu().numpy(),
        query_labels.cpu().numpy(),
        gallery_labels.cpu().numpy(),
        metric=metric,
        ranks=ranks,
    )
