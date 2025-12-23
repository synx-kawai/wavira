"""
Crowd Level Dataset for WiFi CSI Data

Issue #5: 混雑レベル推定モデルの開発

Loads HDF5 files created by data_collector.py and prepares
them for training crowd level estimation models.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


class CrowdDataset(Dataset):
    """
    Dataset for crowd level estimation from CSI data.

    Loads multiple HDF5 files and creates sliding window samples
    for training.

    Args:
        data_files: List of HDF5 file paths or directory containing HDF5 files
        window_size: Number of time steps per sample
        stride: Stride for sliding window (default: window_size // 2)
        mode: "regression" for person count, "classification" for crowd level
        normalize: Whether to normalize amplitude values
        augment: Whether to apply data augmentation
    """

    # Crowd level thresholds for classification
    CROWD_LEVELS = {
        0: "empty",     # 0 people
        1: "low",       # 1-2 people
        2: "medium",    # 3-5 people
        3: "high",      # 6+ people
    }

    def __init__(
        self,
        data_files: Union[str, Path, List[str], List[Path]],
        window_size: int = 100,
        stride: Optional[int] = None,
        mode: str = "regression",
        normalize: bool = True,
        augment: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride or window_size // 2
        self.mode = mode
        self.normalize = normalize
        self.augment = augment

        # Collect all HDF5 files
        self.files = self._collect_files(data_files)
        if not self.files:
            raise ValueError(f"No HDF5 files found in {data_files}")

        logger.info(f"Found {len(self.files)} HDF5 files")

        # Load all data
        self.samples = []
        self.labels = []
        self._load_data()

        # Compute normalization statistics
        if self.normalize:
            self._compute_stats()

    def _collect_files(
        self,
        data_files: Union[str, Path, List[str], List[Path]]
    ) -> List[Path]:
        """Collect all HDF5 files from input."""
        if isinstance(data_files, (str, Path)):
            path = Path(data_files)
            if path.is_file():
                return [path]
            elif path.is_dir():
                return sorted(path.glob("*.h5")) + sorted(path.glob("*.hdf5"))
            else:
                raise ValueError(f"Path does not exist: {path}")
        else:
            return [Path(f) for f in data_files]

    def _load_data(self):
        """Load and process all HDF5 files."""
        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as f:
                    amplitudes = f['amplitudes'][:]
                    num_people = f.attrs['num_people']

                    # Create sliding window samples
                    n_samples = len(amplitudes)
                    for start in range(0, n_samples - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        window = amplitudes[start:end]

                        self.samples.append(window)
                        self.labels.append(num_people)

                logger.info(f"Loaded {file_path.name}: {num_people} people, "
                          f"{len(amplitudes)} samples")

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not self.samples:
            raise ValueError("No valid samples loaded from files")

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        logger.info(f"Total samples: {len(self.samples)}")

    def _compute_stats(self):
        """Compute mean and std for normalization."""
        self.mean = np.mean(self.samples)
        self.std = np.std(self.samples) + 1e-8

    def _to_crowd_level(self, num_people: int) -> int:
        """Convert person count to crowd level class."""
        if num_people == 0:
            return 0  # empty
        elif num_people <= 2:
            return 1  # low
        elif num_people <= 5:
            return 2  # medium
        else:
            return 3  # high

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.02, x.shape)
            x = x + noise

        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        # Random time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-5, 6)
            x = np.roll(x, shift, axis=0)

        return x

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.samples[idx].copy()
        label = self.labels[idx]

        # Data augmentation
        if self.augment:
            x = self._augment(x)

        # Normalize
        if self.normalize:
            x = (x - self.mean) / self.std

        # Convert to tensors
        x = torch.from_numpy(x).float()

        if self.mode == "classification":
            y = torch.tensor(self._to_crowd_level(int(label))).long()
        else:
            y = torch.tensor(label).float().unsqueeze(0)

        return x, y

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset."""
        if self.mode != "classification":
            raise ValueError("Class weights only available for classification mode")

        class_labels = [self._to_crowd_level(int(l)) for l in self.labels]
        counts = np.bincount(class_labels, minlength=4)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * len(self.CROWD_LEVELS)
        return torch.from_numpy(weights).float()

    @staticmethod
    def get_dataloader(
        data_files: Union[str, Path, List[str], List[Path]],
        batch_size: int = 32,
        window_size: int = 100,
        mode: str = "regression",
        shuffle: bool = True,
        num_workers: int = 0,
        augment: bool = False,
    ) -> DataLoader:
        """Create a DataLoader from data files."""
        dataset = CrowdDataset(
            data_files=data_files,
            window_size=window_size,
            mode=mode,
            augment=augment,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


class SyntheticCrowdDataset(Dataset):
    """
    Synthetic dataset for testing model architecture.

    Generates fake CSI data with crowd-level-dependent patterns.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        window_size: int = 100,
        n_subcarriers: int = 52,
        max_people: int = 10,
        mode: str = "regression",
    ):
        self.n_samples = n_samples
        self.window_size = window_size
        self.n_subcarriers = n_subcarriers
        self.max_people = max_people
        self.mode = mode

        self.samples, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic CSI data."""
        samples = []
        labels = []

        for _ in range(self.n_samples):
            # Random number of people
            num_people = np.random.randint(0, self.max_people + 1)

            # Base signal
            base = np.random.uniform(40, 60, (self.window_size, self.n_subcarriers))

            # Add person-dependent variations
            for p in range(num_people):
                # Each person adds variance and periodic movement
                freq = np.random.uniform(0.1, 0.5)  # Movement frequency
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(2, 5)

                t = np.arange(self.window_size) / 10.0
                movement = amplitude * np.sin(2 * np.pi * freq * t + phase)

                # Add to random subcarriers
                affected = np.random.choice(self.n_subcarriers, size=10, replace=False)
                for sc in affected:
                    base[:, sc] += movement

            # Add noise
            noise = np.random.normal(0, 1, base.shape)
            signal = base + noise

            samples.append(signal.astype(np.float32))
            labels.append(num_people)

        return np.array(samples), np.array(labels, dtype=np.float32)

    def _to_crowd_level(self, num_people: int) -> int:
        """Convert person count to crowd level class."""
        if num_people == 0:
            return 0
        elif num_people <= 2:
            return 1
        elif num_people <= 5:
            return 2
        else:
            return 3

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.samples[idx]
        label = self.labels[idx]

        # Normalize
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = torch.from_numpy(x).float()

        if self.mode == "classification":
            y = torch.tensor(self._to_crowd_level(int(label))).long()
        else:
            y = torch.tensor(label).float().unsqueeze(0)

        return x, y
