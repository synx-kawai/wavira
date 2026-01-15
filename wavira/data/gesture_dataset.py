"""
Gesture Recognition Dataset

Dataset classes for loading and processing CSI data for gesture recognition.
Supports sliding window extraction and data augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Union
import json
import h5py


class GestureDataset(Dataset):
    """
    Dataset for gesture recognition from CSI data.

    Handles loading CSI sequences from various formats and extracting
    fixed-size windows for training 3D CNN models.

    Args:
        data_dir: Directory containing gesture data
        gesture_labels: List of gesture class names (order matters)
        n_frames: Number of frames per sample (default: 32)
        frame_stride: Stride for sliding window extraction (default: 8)
        transform: Optional transform to apply to samples
        preprocess_fn: Optional preprocessing function
        split: Data split ('train', 'val', 'test', or None for all)
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        gesture_labels: Optional[List[str]] = None,
        n_frames: int = 32,
        frame_stride: int = 8,
        transform: Optional[Callable] = None,
        preprocess_fn: Optional[Callable] = None,
        split: Optional[str] = None,
    ):
        self.n_frames = n_frames
        self.frame_stride = frame_stride
        self.transform = transform
        self.preprocess_fn = preprocess_fn

        # Default gesture labels from the paper
        self.gesture_labels = gesture_labels or [
            "zoom_out",
            "zoom_in",
            "circle_left",
            "circle_right",
            "swipe_left",
            "swipe_right",
            "flip_up",
            "flip_down",
        ]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.gesture_labels)}

        # Storage for samples
        self.samples: List[Tuple[np.ndarray, int]] = []

        if data_dir is not None:
            self.load_from_directory(Path(data_dir), split)

    def load_from_directory(self, data_dir: Path, split: Optional[str] = None):
        """
        Load gesture data from directory structure.

        Expected structure:
        data_dir/
            gesture_label/
                sample_001.npy
                sample_002.npy
                ...

        Or:
        data_dir/
            train/
                gesture_label/...
            val/
                gesture_label/...
        """
        data_dir = Path(data_dir)

        if split and (data_dir / split).exists():
            data_dir = data_dir / split

        for gesture_label in self.gesture_labels:
            gesture_dir = data_dir / gesture_label
            if not gesture_dir.exists():
                continue

            label_idx = self.label_to_idx[gesture_label]

            for npy_file in sorted(gesture_dir.glob("*.npy")):
                csi_data = np.load(npy_file)
                self._extract_windows(csi_data, label_idx)

    def _extract_windows(self, csi_data: np.ndarray, label_idx: int):
        """
        Extract sliding windows from a CSI sequence.

        Args:
            csi_data: CSI data of shape (n_routes, n_subcarriers, n_packets)
            label_idx: Gesture label index
        """
        # Apply preprocessing if provided
        if self.preprocess_fn is not None:
            csi_data = self.preprocess_fn(csi_data)

        n_packets = csi_data.shape[-1]

        # Extract windows with sliding stride
        start = 0
        while start + self.n_frames <= n_packets:
            window = csi_data[..., start:start + self.n_frames].copy()
            self.samples.append((window, label_idx))
            start += self.frame_stride

    def add_samples(
        self,
        csi_data: np.ndarray,
        label: Union[int, str],
        extract_windows: bool = True
    ):
        """
        Add samples to the dataset programmatically.

        Args:
            csi_data: CSI data array
            label: Gesture label (index or string)
            extract_windows: Whether to extract sliding windows
        """
        if isinstance(label, str):
            label_idx = self.label_to_idx[label]
        else:
            label_idx = label

        if extract_windows:
            self._extract_windows(csi_data, label_idx)
        else:
            if self.preprocess_fn is not None:
                csi_data = self.preprocess_fn(csi_data)
            self.samples.append((csi_data, label_idx))

    def load_from_hdf5(self, hdf5_path: Union[str, Path], split: Optional[str] = None):
        """
        Load gesture data from HDF5 file.

        Expected HDF5 structure:
        /gesture_label/sample_idx/csi -> CSI data
        /gesture_label/sample_idx/metadata -> JSON metadata

        Or with splits:
        /train/gesture_label/...
        /val/gesture_label/...
        """
        with h5py.File(hdf5_path, 'r') as f:
            root = f[split] if split and split in f else f

            for gesture_label in self.gesture_labels:
                if gesture_label not in root:
                    continue

                label_idx = self.label_to_idx[gesture_label]
                gesture_group = root[gesture_label]

                for sample_key in gesture_group.keys():
                    csi_data = gesture_group[sample_key]['csi'][:]
                    self._extract_windows(csi_data, label_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        csi_data, label = self.samples[idx]

        # Convert to tensor
        csi_tensor = torch.from_numpy(csi_data).float()

        # Apply transform if provided
        if self.transform is not None:
            csi_tensor = self.transform(csi_tensor)

        return csi_tensor, label

    def get_label_counts(self) -> Dict[str, int]:
        """Get count of samples per gesture class."""
        counts = {label: 0 for label in self.gesture_labels}
        for _, label_idx in self.samples:
            counts[self.gesture_labels[label_idx]] += 1
        return counts


class DualDeviceGestureDataset(Dataset):
    """
    Dataset for gesture recognition from dual ESP32 CSI data.

    Handles synchronized CSI data from two ESP32 devices.

    Args:
        data_dir: Directory containing synchronized dual-device data
        gesture_labels: List of gesture class names
        n_frames: Number of frames per sample
        frame_stride: Stride for sliding window extraction
        transform: Optional transform to apply
        preprocess_fn: Optional preprocessing function
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        gesture_labels: Optional[List[str]] = None,
        n_frames: int = 32,
        frame_stride: int = 8,
        transform: Optional[Callable] = None,
        preprocess_fn: Optional[Callable] = None,
    ):
        self.n_frames = n_frames
        self.frame_stride = frame_stride
        self.transform = transform
        self.preprocess_fn = preprocess_fn

        self.gesture_labels = gesture_labels or [
            "zoom_out", "zoom_in", "circle_left", "circle_right",
            "swipe_left", "swipe_right", "flip_up", "flip_down",
        ]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.gesture_labels)}

        # Storage: (device1_csi, device2_csi, label)
        self.samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

        if data_dir is not None:
            self.load_from_directory(Path(data_dir))

    def load_from_directory(self, data_dir: Path):
        """
        Load dual-device data from directory.

        Expected structure:
        data_dir/
            gesture_label/
                sample_001_device1.npy
                sample_001_device2.npy
                sample_002_device1.npy
                sample_002_device2.npy
        """
        data_dir = Path(data_dir)

        for gesture_label in self.gesture_labels:
            gesture_dir = data_dir / gesture_label
            if not gesture_dir.exists():
                continue

            label_idx = self.label_to_idx[gesture_label]

            # Find paired samples
            device1_files = sorted(gesture_dir.glob("*_device1.npy"))

            for d1_file in device1_files:
                d2_file = d1_file.with_name(d1_file.name.replace("_device1", "_device2"))

                if not d2_file.exists():
                    continue

                csi_d1 = np.load(d1_file)
                csi_d2 = np.load(d2_file)

                self._extract_windows(csi_d1, csi_d2, label_idx)

    def _extract_windows(
        self,
        csi_d1: np.ndarray,
        csi_d2: np.ndarray,
        label_idx: int
    ):
        """Extract synchronized windows from both devices."""
        if self.preprocess_fn is not None:
            csi_d1 = self.preprocess_fn(csi_d1)
            csi_d2 = self.preprocess_fn(csi_d2)

        # Use minimum length
        n_packets = min(csi_d1.shape[-1], csi_d2.shape[-1])

        start = 0
        while start + self.n_frames <= n_packets:
            window_d1 = csi_d1[..., start:start + self.n_frames].copy()
            window_d2 = csi_d2[..., start:start + self.n_frames].copy()
            self.samples.append((window_d1, window_d2, label_idx))
            start += self.frame_stride

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        csi_d1, csi_d2, label = self.samples[idx]

        tensor_d1 = torch.from_numpy(csi_d1).float()
        tensor_d2 = torch.from_numpy(csi_d2).float()

        if self.transform is not None:
            tensor_d1 = self.transform(tensor_d1)
            tensor_d2 = self.transform(tensor_d2)

        return tensor_d1, tensor_d2, label


class SyntheticGestureDataset(Dataset):
    """
    Synthetic gesture dataset for testing and development.

    Generates artificial CSI patterns that simulate gesture-induced
    signal variations.
    """

    def __init__(
        self,
        n_samples_per_gesture: int = 100,
        n_frames: int = 32,
        n_routes: int = 3,
        n_subcarriers: int = 114,
        noise_level: float = 0.1,
        gesture_labels: Optional[List[str]] = None,
    ):
        self.n_frames = n_frames
        self.n_routes = n_routes
        self.n_subcarriers = n_subcarriers
        self.noise_level = noise_level

        self.gesture_labels = gesture_labels or [
            "zoom_out", "zoom_in", "circle_left", "circle_right",
            "swipe_left", "swipe_right", "flip_up", "flip_down",
        ]
        self.n_gestures = len(self.gesture_labels)

        # Generate samples
        self.samples = []
        self._generate_samples(n_samples_per_gesture)

    def _generate_samples(self, n_samples_per_gesture: int):
        """Generate synthetic gesture samples."""
        # Use local RNG to avoid affecting global random state
        self._rng = np.random.default_rng(42)

        for label_idx, gesture_label in enumerate(self.gesture_labels):
            for _ in range(n_samples_per_gesture):
                csi = self._generate_gesture_csi(gesture_label)
                self.samples.append((csi, label_idx))

    def _generate_gesture_csi(self, gesture_type: str) -> np.ndarray:
        """
        Generate synthetic CSI for a specific gesture type.

        Different gestures create different temporal patterns in CSI.
        """
        t = np.linspace(0, 1, self.n_frames)
        csi = np.zeros((self.n_routes, self.n_subcarriers, self.n_frames))

        # Base amplitude pattern
        base = self._rng.uniform(20, 40, (self.n_routes, self.n_subcarriers, 1))

        # Gesture-specific patterns
        if gesture_type == "zoom_out":
            # Expanding motion - amplitude increases then decreases
            pattern = np.sin(np.pi * t) * 10
        elif gesture_type == "zoom_in":
            # Contracting motion - amplitude decreases then increases
            pattern = -np.sin(np.pi * t) * 10
        elif gesture_type in ["circle_left", "circle_right"]:
            # Circular motion - sinusoidal pattern
            sign = 1 if gesture_type == "circle_left" else -1
            pattern = sign * np.sin(2 * np.pi * t) * 8
        elif gesture_type == "swipe_left":
            # Linear left motion
            pattern = -t * 15
        elif gesture_type == "swipe_right":
            # Linear right motion
            pattern = t * 15
        elif gesture_type == "flip_up":
            # Quick upward motion
            pattern = np.where(t < 0.5, t * 20, (1 - t) * 20)
        elif gesture_type == "flip_down":
            # Quick downward motion
            pattern = np.where(t < 0.5, -t * 20, -(1 - t) * 20)
        else:
            pattern = np.zeros_like(t)

        # Apply pattern with subcarrier variation
        for sc in range(self.n_subcarriers):
            phase_shift = sc * 0.1
            csi[:, sc, :] = base[:, sc, :] + pattern * np.cos(phase_shift)

        # Add noise
        noise = self._rng.normal(0, self.noise_level * base.mean(), csi.shape)
        csi += noise

        return csi.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        csi, label = self.samples[idx]
        return torch.from_numpy(csi), label


def create_gesture_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    n_frames: int = 32,
    frame_stride: int = 8,
    preprocess_fn: Optional[Callable] = None,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for gesture recognition.

    Args:
        data_dir: Directory containing gesture data
        batch_size: Batch size for training
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        n_frames: Number of frames per sample
        frame_stride: Stride for sliding window
        preprocess_fn: Optional preprocessing function
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, random_split

    # Create full dataset
    dataset = GestureDataset(
        data_dir=data_dir,
        n_frames=n_frames,
        frame_stride=frame_stride,
        preprocess_fn=preprocess_fn,
    )

    # Split dataset
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
