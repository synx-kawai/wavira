#!/usr/bin/env python3
"""
Training Script for Crowd Level Estimation Model

Issue #5: 混雑レベル推定モデルの開発

Usage:
    # Train with synthetic data (for testing)
    python train_crowd.py --synthetic

    # Train with real data
    python train_crowd.py --data_dir data/crowd_csi/

    # Train classification model
    python train_crowd.py --data_dir data/crowd_csi/ --mode classification
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavira.models.crowd_estimator import CrowdEstimator, CrowdEstimatorConfig, create_model
from wavira.data.crowd_dataset import CrowdDataset, SyntheticCrowdDataset

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False
    SummaryWriter = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str = "regression",
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        if mode == "regression":
            loss = criterion(outputs, batch_y)
        else:
            loss = criterion(outputs, batch_y)
            pred = outputs.argmax(dim=1)
            correct += (pred == batch_y).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = {
        "loss": total_loss / total_samples,
    }
    if mode == "classification":
        metrics["accuracy"] = correct / total_samples

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str = "regression",
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)

        if mode == "regression":
            loss = criterion(outputs, batch_y)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(batch_y.squeeze().cpu().numpy())
        else:
            loss = criterion(outputs, batch_y)
            pred = outputs.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    metrics = {
        "loss": total_loss / total_samples,
    }

    if mode == "regression":
        import numpy as np
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        mae = np.abs(preds - labels).mean()
        rmse = np.sqrt(((preds - labels) ** 2).mean())
        metrics["mae"] = mae
        metrics["rmse"] = rmse
    else:
        metrics["accuracy"] = correct / total_samples

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Crowd Estimator Model")

    # Data
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing HDF5 data files")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--window_size", type=int, default=100,
                        help="Window size (time steps)")

    # Model
    parser.add_argument("--mode", type=str, default="regression",
                        choices=["regression", "classification"],
                        help="Model mode")
    parser.add_argument("--encoder", type=str, default="transformer",
                        choices=["transformer", "lstm", "bilstm"],
                        help="Encoder type")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of encoder layers")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/crowd",
                        help="Output directory for checkpoints")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Experiment name")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name or f"{args.mode}_{args.encoder}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard
    writer = None
    if TENSORBOARD:
        log_dir = output_dir / "logs"
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logs: {log_dir}")

    # Load data
    if args.synthetic:
        logger.info("Using synthetic data for testing")
        full_dataset = SyntheticCrowdDataset(
            n_samples=2000,
            window_size=args.window_size,
            n_subcarriers=52,
            max_people=10,
            mode=args.mode,
        )
    else:
        if args.data_dir is None:
            logger.error("Please specify --data_dir or use --synthetic")
            sys.exit(1)
        logger.info(f"Loading data from {args.data_dir}")
        full_dataset = CrowdDataset(
            data_files=args.data_dir,
            window_size=args.window_size,
            mode=args.mode,
            augment=True,
        )

    # Split dataset
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    config = CrowdEstimatorConfig(
        n_subcarriers=52,
        seq_length=args.window_size,
        encoder_type=args.encoder,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        mode=args.mode,
    )
    model = CrowdEstimator(config).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    if args.mode == "regression":
        criterion = nn.MSELoss()
    else:
        if hasattr(full_dataset, 'get_class_weights'):
            weights = full_dataset.get_class_weights().to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    best_val_metric = float("inf") if args.mode == "regression" else 0.0
    patience_counter = 0
    max_patience = 20

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args.mode)

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, args.mode)

        # Update scheduler
        scheduler.step()

        # Logging
        if args.mode == "regression":
            logger.info(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.2f}, "
                f"Val RMSE: {val_metrics['rmse']:.2f}"
            )
            current_metric = val_metrics["mae"]
            is_best = current_metric < best_val_metric
        else:
            logger.info(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2%}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2%}"
            )
            current_metric = val_metrics["accuracy"]
            is_best = current_metric > best_val_metric

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            if args.mode == "regression":
                writer.add_scalar("MAE/val", val_metrics["mae"], epoch)
                writer.add_scalar("RMSE/val", val_metrics["rmse"], epoch)
            else:
                writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
                writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

        # Save best model
        if is_best:
            best_val_metric = current_metric
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            logger.info(f"  -> Saved best model")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 20 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Early stopping
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final save
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    if writer:
        writer.close()

    logger.info(f"Training complete. Best {'MAE' if args.mode == 'regression' else 'Accuracy'}: "
                f"{best_val_metric:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
