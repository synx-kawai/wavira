"""
Training Pipeline for WhoFi Model

Implements the training procedure as described in the paper:
- Adam optimizer with lr=0.0001
- StepLR scheduler (factor=0.95 every 50 epochs)
- 300 training epochs
- Batch size of 8
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from wavira.models.whofi import WhoFi
from wavira.losses.inbatch_loss import InBatchNegativeLoss
from wavira.utils.metrics import evaluate_reid_torch


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model config
    n_channels: int = 3
    n_subcarriers: int = 114
    encoder_type: str = "transformer"
    hidden_dim: int = 256
    signature_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    nhead: int = 8

    # Training config
    batch_size: int = 8
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    epochs: int = 300
    lr_step_size: int = 50
    lr_gamma: float = 0.95
    temperature: float = 0.07

    # Data config
    sequence_length: int = 200

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    eval_interval: int = 10
    save_interval: int = 50

    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001


class Trainer:
    """
    Trainer for WhoFi person re-identification model.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model: Optional[WhoFi] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model: Pre-initialized model (optional)
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        # Initialize model
        if model is not None:
            self.model = model
        else:
            self.model = WhoFi(
                n_channels=self.config.n_channels,
                n_subcarriers=self.config.n_subcarriers,
                encoder_type=self.config.encoder_type,
                hidden_dim=self.config.hidden_dim,
                signature_dim=self.config.signature_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                nhead=self.config.nhead,
            )

        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = InBatchNegativeLoss(
            temperature=self.config.temperature,
            symmetric=True,
        )

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )

        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.best_rank1 = 0.0
        self.training_history: List[Dict[str, float]] = []

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            signatures = self.model(data)

            # Compute loss
            loss = self.criterion(signatures, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / n_batches

        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(
        self,
        query_loader: DataLoader,
        gallery_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            query_loader: Query data loader
            gallery_loader: Gallery data loader (if None, uses query_loader for both)

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        # Extract query features
        query_features = []
        query_labels = []

        for data, labels in query_loader:
            data = data.to(self.device)
            signatures = self.model(data)
            query_features.append(signatures.cpu())
            query_labels.append(labels)

        query_features = torch.cat(query_features, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        # Extract gallery features
        if gallery_loader is None:
            gallery_features = query_features
            gallery_labels = query_labels
        else:
            gallery_features = []
            gallery_labels = []

            for data, labels in gallery_loader:
                data = data.to(self.device)
                signatures = self.model(data)
                gallery_features.append(signatures.cpu())
                gallery_labels.append(labels)

            gallery_features = torch.cat(gallery_features, dim=0)
            gallery_labels = torch.cat(gallery_labels, dim=0)

        # Compute metrics
        metrics = evaluate_reid_torch(
            query_features,
            gallery_features,
            query_labels,
            gallery_labels,
            metric="cosine",
            ranks=(1, 5, 10),
        )

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            callbacks: Optional list of callback functions

        Returns:
            Training history and best metrics
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        patience_counter = 0
        best_epoch = 0

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Update scheduler
            self.scheduler.step()

            epoch_time = time.time() - start_time

            # Log training metrics
            log_entry = {
                "epoch": epoch + 1,
                "lr": self.scheduler.get_last_lr()[0],
                "epoch_time": epoch_time,
                **train_metrics,
            }

            # Evaluate periodically
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                log_entry.update({f"val_{k}": v for k, v in val_metrics.items()})

                # Check for improvement
                current_map = val_metrics["mAP"]
                current_rank1 = val_metrics["Rank-1"]

                if current_map > self.best_map + self.config.early_stopping_min_delta:
                    self.best_map = current_map
                    self.best_rank1 = current_rank1
                    best_epoch = epoch + 1
                    patience_counter = 0

                    # Save best model
                    self.save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1

                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Loss: {train_metrics['train_loss']:.4f} | "
                    f"mAP: {val_metrics['mAP']:.4f} | "
                    f"Rank-1: {val_metrics['Rank-1']:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            else:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Loss: {train_metrics['train_loss']:.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            self.training_history.append(log_entry)

            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, log_entry)

        # Save final model
        self.save_checkpoint("final_model.pt")

        return {
            "history": self.training_history,
            "best_map": self.best_map,
            "best_rank1": self.best_rank1,
            "best_epoch": best_epoch,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_map": self.best_map,
                "best_rank1": self.best_rank1,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_map = checkpoint["best_map"]
        self.best_rank1 = checkpoint["best_rank1"]

        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
