#!/usr/bin/env python3
"""
Training script for WhoFi/Wavira person re-identification.

Usage:
    python scripts/train.py --data_dir /path/to/data
    python scripts/train.py --use_synthetic  # Use synthetic data for testing
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavira.models.whofi import WhoFi
from wavira.data.dataset import CSIDataset, create_synthetic_csi_data
from wavira.training.trainer import Trainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train WhoFi model")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing CSI data",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )

    # Model arguments
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="transformer",
        choices=["transformer", "lstm", "bilstm"],
        help="Encoder architecture type",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--signature_dim",
        type=int,
        default=256,
        help="Signature vector dimension",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads (transformer only)",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=200,
        help="Sequence length (number of packets)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for in-batch negative loss",
    )

    # Misc arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load or create data
    if args.use_synthetic:
        print("Creating synthetic data for testing...")
        data, labels = create_synthetic_csi_data(
            n_persons=14,
            samples_per_person=60,
            n_rx_antennas=3,
            n_subcarriers=114,
            n_packets=500,  # Shorter for faster testing
            seed=args.seed,
        )
        n_channels = 3
        n_subcarriers = 114

        # Create dataset
        dataset = CSIDataset(
            data=data,
            labels=labels,
            sequence_length=args.sequence_length,
            preprocess=False,  # Synthetic data is already processed
            normalize=True,
        )

    elif args.data_dir:
        print(f"Loading data from {args.data_dir}")
        dataset = CSIDataset(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            preprocess=True,
            normalize=True,
        )
        # Infer dimensions from first sample
        sample, _ = dataset[0]
        n_channels = sample.shape[0]
        n_subcarriers = sample.shape[1]

    else:
        print("Error: Must provide --data_dir or --use_synthetic")
        sys.exit(1)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of classes: {dataset.n_classes}")
    print(f"Input shape: ({n_channels}, {n_subcarriers}, {args.sequence_length})")

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False,
    )

    # Create training config
    config = TrainingConfig(
        n_channels=n_channels,
        n_subcarriers=n_subcarriers,
        encoder_type=args.encoder_type,
        hidden_dim=args.hidden_dim,
        signature_dim=args.signature_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        temperature=args.temperature,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        eval_interval=10,
        save_interval=50,
    )

    # Create trainer
    trainer = Trainer(config)

    print("\nModel architecture:")
    print(trainer.model)

    # Train
    print("\nStarting training...")
    results = trainer.train(train_loader, val_loader)

    # Print final results
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best mAP: {results['best_map']:.4f}")
    print(f"Best Rank-1: {results['best_rank1']:.4f}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
