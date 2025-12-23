"""
CSI Dataset Module

Provides PyTorch Dataset classes for loading and processing
Wi-Fi Channel State Information data for person re-identification.
"""

import os
import logging
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Any, Callable

from wavira.data.preprocessing import preprocess_csi, normalize_csi

logger = logging.getLogger(__name__)


class CSIDataset(Dataset):
    """
    Dataset for CSI-based person re-identification.

    Supports loading from numpy arrays, from directory structure, or from file list.
    Expected directory structure:
        data_dir/
            person_001/
                sample_001.npy
                sample_002.npy
                ...
            person_002/
                ...

    Or file list format (each line contains path and optional label):
        /path/to/sample1.npy
        /path/to/sample2.npy
        ...

    Each .npy file should contain CSI data in format:
    - Complex: (rx_antennas, subcarriers, packets)
    - Or preprocessed: (subcarriers, packets) or (channels, subcarriers, packets)
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        data_dir: Optional[str] = None,
        file_list: Optional[str] = None,
        samples_per_class: int = 500,
        sequence_length: int = 200,
        preprocess: bool = True,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        use_phase: bool = False,
    ):
        """
        Initialize CSI Dataset.

        Args:
            data: CSI data array of shape (n_samples, ..., packets)
            labels: Person identity labels of shape (n_samples,)
            data_dir: Directory containing CSI data organized by person
            file_list: Path to file containing list of .npy file paths
            samples_per_class: Number of samples per class (for file_list mode)
            sequence_length: Number of packets to use per sample
            preprocess: Whether to apply CSI preprocessing
            normalize: Whether to normalize data
            transform: Optional transform function
            use_phase: Whether to include phase information
        """
        self.sequence_length = sequence_length
        self.preprocess = preprocess
        self.normalize = normalize
        self.transform = transform
        self.use_phase = use_phase
        self.samples_per_class = samples_per_class

        if data is not None and labels is not None:
            self.data = data
            self.labels = labels
            self.n_samples = len(data)
        elif file_list is not None:
            self.data, self.labels = self._load_from_file_list(file_list)
            self.n_samples = len(self.data)
        elif data_dir is not None:
            self.data, self.labels = self._load_from_directory(data_dir)
            self.n_samples = len(self.data)
        else:
            raise ValueError("Must provide either (data, labels), file_list, or data_dir")

        # Create label to index mapping
        unique_labels = np.unique(self.labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.n_classes = len(unique_labels)

    def _load_from_directory(self, data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load CSI data from directory structure."""
        data = []
        labels = []

        persons = sorted(os.listdir(data_dir))
        for person_id in persons:
            person_path = os.path.join(data_dir, person_id)
            if not os.path.isdir(person_path):
                continue

            samples = sorted(os.listdir(person_path))
            for sample_file in samples:
                if not sample_file.endswith('.npy'):
                    continue

                sample_path = os.path.join(person_path, sample_file)
                sample_data = np.load(sample_path)
                data.append(sample_data)
                labels.append(person_id)

        return data, np.array(labels)

    def _load_from_file_list(self, file_list_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load CSI data from a file list.

        File format: one file path per line.
        Labels are generated based on file index and samples_per_class.
        """
        data = []
        labels = []
        missing_files = []

        # Read file list
        with open(file_list_path, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]

        logger.info(f"Loading {len(file_paths)} files from {file_list_path}")

        # Load files and assign labels based on index
        for idx, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                continue

            sample_data = np.load(file_path)
            data.append(sample_data)

            # Assign label based on file index
            label = idx // self.samples_per_class
            labels.append(label)

        # Log summary of missing files
        if missing_files:
            logger.warning(
                f"Skipped {len(missing_files)} missing files. "
                f"First few: {missing_files[:3]}"
            )

        logger.info(f"Successfully loaded {len(data)} samples")

        return data, np.array(labels)

    def _process_sample(self, sample: np.ndarray) -> torch.Tensor:
        """Process a single CSI sample."""
        # Apply preprocessing if needed
        if self.preprocess and np.iscomplexobj(sample):
            sample = preprocess_csi(sample, use_phase=self.use_phase)

        # Ensure correct shape: (channels, subcarriers, packets)
        if sample.ndim == 2:
            sample = sample[np.newaxis, ...]

        # Handle sequence length
        n_packets = sample.shape[-1]
        if n_packets > self.sequence_length:
            # Random crop
            start = np.random.randint(0, n_packets - self.sequence_length)
            sample = sample[..., start:start + self.sequence_length]
        elif n_packets < self.sequence_length:
            # Pad with zeros
            pad_width = [(0, 0)] * (sample.ndim - 1) + [(0, self.sequence_length - n_packets)]
            sample = np.pad(sample, pad_width, mode='constant')

        # Normalize
        if self.normalize:
            sample, _, _ = normalize_csi(sample, axis=-1)

        # Convert to tensor
        tensor = torch.from_numpy(sample).float()

        # Apply custom transform
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample by index.

        Returns:
            Tuple of (csi_tensor, label_index)
        """
        if isinstance(self.data, list):
            sample = self.data[idx]
        else:
            sample = self.data[idx]

        label = self.labels[idx]
        label_idx = self.label_to_idx[label]

        tensor = self._process_sample(sample)

        return tensor, label_idx


class CSIPairDataset(Dataset):
    """
    Dataset that returns pairs of samples for contrastive learning.

    Each item returns (anchor, positive, negative) triplets or
    (sample1, sample2, is_same) pairs.
    """

    def __init__(
        self,
        csi_dataset: CSIDataset,
        mode: str = "triplet",
    ):
        """
        Initialize pair dataset.

        Args:
            csi_dataset: Base CSI dataset
            mode: 'triplet' for (anchor, positive, negative) or 'pair' for (s1, s2, label)
        """
        self.dataset = csi_dataset
        self.mode = mode

        # Build index mapping for each person
        self.label_indices: Dict[Any, List[int]] = {}
        for idx in range(len(self.dataset)):
            label = self.dataset.labels[idx]
            if label not in self.label_indices:
                self.label_indices[label] = []
            self.label_indices[label].append(idx)

        self.labels = list(self.label_indices.keys())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        anchor, anchor_label = self.dataset[idx]
        anchor_person = self.dataset.labels[idx]

        # Get positive sample (same person, different sample)
        positive_indices = [i for i in self.label_indices[anchor_person] if i != idx]
        if positive_indices:
            positive_idx = np.random.choice(positive_indices)
        else:
            positive_idx = idx
        positive, _ = self.dataset[positive_idx]

        if self.mode == "triplet":
            # Get negative sample (different person)
            negative_labels = [l for l in self.labels if l != anchor_person]
            negative_person = np.random.choice(negative_labels)
            negative_idx = np.random.choice(self.label_indices[negative_person])
            negative, _ = self.dataset[negative_idx]

            return anchor, positive, negative, anchor_label

        else:  # pair mode
            # Return positive or negative pair randomly
            if np.random.random() > 0.5:
                return anchor, positive, torch.tensor(1)
            else:
                negative_labels = [l for l in self.labels if l != anchor_person]
                negative_person = np.random.choice(negative_labels)
                negative_idx = np.random.choice(self.label_indices[negative_person])
                negative, _ = self.dataset[negative_idx]
                return anchor, negative, torch.tensor(0)


def create_synthetic_csi_data(
    n_persons: int = 14,
    samples_per_person: int = 60,
    n_rx_antennas: int = 3,
    n_subcarriers: int = 114,
    n_packets: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic CSI data for testing.

    Each person has a unique base pattern that is slightly modified
    for each sample to simulate real-world variation.

    Args:
        n_persons: Number of unique persons
        samples_per_person: Number of samples per person
        n_rx_antennas: Number of receiver antennas
        n_subcarriers: Number of subcarriers
        n_packets: Number of packets per sample
        seed: Random seed

    Returns:
        Tuple of (data, labels) where data has shape
        (n_samples, n_rx_antennas, n_subcarriers, n_packets)
    """
    np.random.seed(seed)

    n_samples = n_persons * samples_per_person
    data = []
    labels = []

    for person_id in range(n_persons):
        # Create unique base pattern for this person
        base_amplitude = np.random.uniform(0.5, 1.5, (n_rx_antennas, n_subcarriers))
        base_freq = np.random.uniform(0.01, 0.1, (n_rx_antennas, n_subcarriers))
        base_phase_offset = np.random.uniform(0, 2 * np.pi, (n_rx_antennas, n_subcarriers))

        for sample_id in range(samples_per_person):
            # Add variation to base pattern
            amplitude_var = base_amplitude + np.random.normal(0, 0.1, base_amplitude.shape)
            freq_var = base_freq + np.random.normal(0, 0.01, base_freq.shape)
            phase_var = base_phase_offset + np.random.normal(0, 0.3, base_phase_offset.shape)

            # Generate time series
            t = np.arange(n_packets)
            csi = np.zeros((n_rx_antennas, n_subcarriers, n_packets), dtype=np.float32)

            for ant in range(n_rx_antennas):
                for sub in range(n_subcarriers):
                    signal = amplitude_var[ant, sub] * np.sin(
                        2 * np.pi * freq_var[ant, sub] * t + phase_var[ant, sub]
                    )
                    # Add noise
                    noise = np.random.normal(0, 0.05, n_packets)
                    csi[ant, sub, :] = signal + noise

            data.append(csi)
            labels.append(person_id)

    return np.array(data), np.array(labels)
