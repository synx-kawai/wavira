#!/usr/bin/env python3
"""
Evaluation script for WhoFi/Wavira person re-identification.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data_dir /path/to/test_data
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavira.models.whofi import WhoFi
from wavira.data.dataset import CSIDataset, create_synthetic_csi_data
from wavira.utils.metrics import evaluate_reid_torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WhoFi model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing test CSI data",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        default=None,
        help="Path to file containing list of .npy files for evaluation",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=500,
        help="Number of samples per class (for file list mode)",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    # SECURITY NOTE: weights_only=False allows arbitrary code execution.
    # Only load checkpoints from trusted sources.
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint["config"]

    # Create model
    model = WhoFi(
        n_channels=config.n_channels,
        n_subcarriers=config.n_subcarriers,
        encoder_type=config.encoder_type,
        hidden_dim=config.hidden_dim,
        signature_dim=config.signature_dim,
        num_layers=config.num_layers,
        dropout=0.0,  # No dropout during evaluation
        nhead=config.nhead,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Training best mAP: {checkpoint['best_map']:.4f}")
    print(f"Training best Rank-1: {checkpoint['best_rank1']:.4f}")

    # Load data
    if args.use_synthetic:
        print("\nCreating synthetic test data...")
        data, labels = create_synthetic_csi_data(
            n_persons=14,
            samples_per_person=20,  # Smaller test set
            n_rx_antennas=config.n_channels,
            n_subcarriers=config.n_subcarriers,
            n_packets=500,
            seed=123,  # Different seed from training
        )
        dataset = CSIDataset(
            data=data,
            labels=labels,
            sequence_length=config.sequence_length,
            preprocess=False,
            normalize=True,
        )
    elif args.file_list:
        print(f"\nLoading test data from {args.file_list}")
        dataset = CSIDataset(
            file_list=args.file_list,
            samples_per_class=args.samples_per_class,
            sequence_length=config.sequence_length,
            preprocess=True,
            normalize=True,
        )
    elif args.data_dir:
        print(f"\nLoading test data from {args.data_dir}")
        dataset = CSIDataset(
            data_dir=args.data_dir,
            sequence_length=config.sequence_length,
            preprocess=True,
            normalize=True,
        )
    else:
        print("Error: Must provide --data_dir, --file_list, or --use_synthetic")
        sys.exit(1)

    print(f"Test dataset size: {len(dataset)} samples")
    print(f"Number of classes: {dataset.n_classes}")

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Extract features
    print("\nExtracting features...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            features = model(data)
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Extracted features shape: {features.shape}")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_reid_torch(
        features, features,  # Query = Gallery for closed-set evaluation
        labels, labels,
        metric="cosine",
        ranks=(1, 5, 10, 20),
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    print(f"mAP:     {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
    print(f"Rank-1:  {metrics['Rank-1']:.4f} ({metrics['Rank-1']*100:.2f}%)")
    print(f"Rank-5:  {metrics['Rank-5']:.4f} ({metrics['Rank-5']*100:.2f}%)")
    print(f"Rank-10: {metrics['Rank-10']:.4f} ({metrics['Rank-10']*100:.2f}%)")
    print(f"Rank-20: {metrics['Rank-20']:.4f} ({metrics['Rank-20']*100:.2f}%)")


if __name__ == "__main__":
    main()
