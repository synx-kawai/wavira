#!/usr/bin/env python3
"""
Gesture Recognition Training Script

Train 3D CNN models for Wi-Fi CSI-based gesture recognition.

Usage:
    # Train with real data
    python scripts/train_gesture.py --data_dir data/gestures --epochs 50

    # Train with synthetic data (for testing)
    python scripts/train_gesture.py --use_synthetic --epochs 25

    # Train with dual ESP32 data
    python scripts/train_gesture.py --data_dir data/dual_gesture --model_type dual
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavira.models.gesture_recognizer import (
    GestureRecognizer3DCNN,
    GestureRecognizerLite,
    DualESP32GestureRecognizer,
    GestureRecognizerConfig,
    DEFAULT_GESTURE_LABELS,
    create_gesture_model,
)
from wavira.data.gesture_dataset import (
    GestureDataset,
    DualDeviceGestureDataset,
    SyntheticGestureDataset,
)
from wavira.data.gesture_preprocessing import GesturePreprocessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train gesture recognition model on CSI data"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing gesture data"
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data for training"
    )
    parser.add_argument(
        "--n_synthetic_samples",
        type=int,
        default=200,
        help="Number of synthetic samples per gesture"
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="standard",
        choices=["standard", "lite", "dual"],
        help="Model architecture type"
    )
    parser.add_argument(
        "--n_gestures",
        type=int,
        default=8,
        help="Number of gesture classes"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=32,
        help="Number of frames per sample"
    )
    parser.add_argument(
        "--n_subcarriers",
        type=int,
        default=114,
        help="Number of CSI subcarriers (ESP32: 114)"
    )
    parser.add_argument(
        "--n_routes",
        type=int,
        default=3,
        help="Number of TX*RX antenna routes"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer type"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate"
    )

    # Preprocessing arguments
    parser.add_argument(
        "--butter_cutoff",
        type=float,
        default=20.0,
        help="Butterworth filter cutoff frequency (Hz)"
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=100.0,
        help="CSI sampling rate (Hz)"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gesture",
        help="Directory to save model and logs"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )

    return parser.parse_args()


def get_device(device_arg: str = None) -> torch.device:
    """Get the appropriate device for training."""
    if device_arg:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_dataloaders(args):
    """Create train/val dataloaders based on arguments."""
    preprocessor = GesturePreprocessor(
        butter_cutoff=args.butter_cutoff,
        sampling_rate=args.sampling_rate,
    )

    if args.use_synthetic:
        logger.info("Creating synthetic gesture dataset...")
        dataset = SyntheticGestureDataset(
            n_samples_per_gesture=args.n_synthetic_samples,
            n_frames=args.n_frames,
            n_routes=args.n_routes,
            n_subcarriers=args.n_subcarriers,
        )
    elif args.model_type == "dual":
        logger.info(f"Loading dual-device data from {args.data_dir}...")
        dataset = DualDeviceGestureDataset(
            data_dir=args.data_dir,
            n_frames=args.n_frames,
            preprocess_fn=preprocessor,
        )
    else:
        if args.data_dir is None:
            raise ValueError("--data_dir is required when not using synthetic data")
        logger.info(f"Loading gesture data from {args.data_dir}...")
        dataset = GestureDataset(
            data_dir=args.data_dir,
            n_frames=args.n_frames,
            preprocess_fn=preprocessor,
        )

    # Split dataset
    total = len(dataset)
    train_size = int(total * 0.8)
    val_size = total - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Dataset split: {train_size} train, {val_size} val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, is_dual=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        if is_dual:
            csi_d1, csi_d2, labels = batch
            csi_d1 = csi_d1.to(device)
            csi_d2 = csi_d2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(csi_d1, csi_d2)
        else:
            csi, labels = batch
            csi = csi.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(csi)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device, is_dual=False):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            if is_dual:
                csi_d1, csi_d2, labels = batch
                csi_d1 = csi_d1.to(device)
                csi_d2 = csi_d2.to(device)
                labels = labels.to(device)
                outputs = model(csi_d1, csi_d2)
            else:
                csi, labels = batch
                csi = csi.to(device)
                labels = labels.to(device)
                outputs = model(csi)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)

    return avg_loss, accuracy


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    config = GestureRecognizerConfig(
        n_gestures=args.n_gestures,
        n_subcarriers=args.n_subcarriers,
        n_routes=args.n_routes,
        n_frames=args.n_frames,
        dropout_rate=args.dropout,
    )

    model = create_gesture_model(
        model_type=args.model_type,
        n_gestures=args.n_gestures,
        n_subcarriers=args.n_subcarriers,
        n_routes=args.n_routes,
        n_frames=args.n_frames,
    )
    model = model.to(device)

    logger.info(f"Model: {args.model_type}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # TensorBoard writer
    writer = SummaryWriter(output_dir / "logs")

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    is_dual = args.model_type == "dual"

    logger.info("Starting training...")

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, is_dual
        )

        # Evaluate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, is_dual
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': {
                    'model_type': args.model_type,
                    'n_gestures': args.n_gestures,
                    'n_subcarriers': args.n_subcarriers,
                    'n_routes': args.n_routes,
                    'n_frames': args.n_frames,
                },
                'gesture_labels': DEFAULT_GESTURE_LABELS[:args.n_gestures],
            }, output_dir / "best_model.pt")

            logger.info(f"New best model saved with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    writer.close()

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
