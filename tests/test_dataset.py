"""
Unit tests for CSI Dataset module.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from wavira.data.dataset import CSIDataset, create_synthetic_csi_data


class TestCSIDataset:
    """Tests for CSIDataset class."""

    def test_load_from_numpy_arrays(self):
        """Test loading dataset from numpy arrays."""
        n_samples = 10
        data = np.random.randn(n_samples, 3, 64, 100).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        dataset = CSIDataset(
            data=data,
            labels=labels,
            sequence_length=50,
            preprocess=False,
            normalize=True,
        )

        assert len(dataset) == n_samples
        assert dataset.n_classes == 5

        sample, label = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape[-1] == 50  # sequence_length

    def test_load_from_file_list(self):
        """Test loading dataset from file list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample files
            file_paths = []
            for i in range(10):
                file_path = os.path.join(tmpdir, f"sample_{i}.npy")
                data = np.random.randn(1, 64, 5).astype(np.complex128)
                np.save(file_path, data)
                file_paths.append(file_path)

            # Create file list
            file_list_path = os.path.join(tmpdir, "files.txt")
            with open(file_list_path, 'w') as f:
                for path in file_paths:
                    f.write(path + '\n')

            # Load dataset
            dataset = CSIDataset(
                file_list=file_list_path,
                samples_per_class=5,
                sequence_length=5,
                preprocess=True,
                normalize=True,
            )

            assert len(dataset) == 10
            assert dataset.n_classes == 2  # 10 samples / 5 per class = 2 classes

    def test_missing_files_handled(self):
        """Test that missing files are logged and skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            file_paths = []
            for i in range(5):
                file_path = os.path.join(tmpdir, f"sample_{i}.npy")
                data = np.random.randn(1, 64, 5).astype(np.complex128)
                np.save(file_path, data)
                file_paths.append(file_path)

            # Add non-existent files
            file_paths.append(os.path.join(tmpdir, "missing_1.npy"))
            file_paths.append(os.path.join(tmpdir, "missing_2.npy"))

            # Create file list
            file_list_path = os.path.join(tmpdir, "files.txt")
            with open(file_list_path, 'w') as f:
                for path in file_paths:
                    f.write(path + '\n')

            # Load dataset - should skip missing files
            dataset = CSIDataset(
                file_list=file_list_path,
                samples_per_class=10,
                sequence_length=5,
                preprocess=True,
                normalize=True,
            )

            assert len(dataset) == 5  # Only 5 valid files

    def test_sequence_length_padding(self):
        """Test that short sequences are padded."""
        n_samples = 5
        n_packets = 10
        sequence_length = 20

        data = np.random.randn(n_samples, 1, 64, n_packets).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2])

        dataset = CSIDataset(
            data=data,
            labels=labels,
            sequence_length=sequence_length,
            preprocess=False,
            normalize=True,
        )

        sample, _ = dataset[0]
        assert sample.shape[-1] == sequence_length

    def test_sequence_length_cropping(self):
        """Test that long sequences are cropped."""
        n_samples = 5
        n_packets = 100
        sequence_length = 20

        data = np.random.randn(n_samples, 1, 64, n_packets).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2])

        dataset = CSIDataset(
            data=data,
            labels=labels,
            sequence_length=sequence_length,
            preprocess=False,
            normalize=True,
        )

        sample, _ = dataset[0]
        assert sample.shape[-1] == sequence_length

    def test_create_synthetic_data(self):
        """Test synthetic data generation."""
        n_persons = 5
        samples_per_person = 10

        data, labels = create_synthetic_csi_data(
            n_persons=n_persons,
            samples_per_person=samples_per_person,
            n_rx_antennas=1,
            n_subcarriers=64,
            n_packets=100,
            seed=42,
        )

        assert data.shape[0] == n_persons * samples_per_person
        assert labels.shape[0] == n_persons * samples_per_person
        assert len(np.unique(labels)) == n_persons

    def test_label_mapping(self):
        """Test that label mapping is correctly created."""
        data = np.random.randn(6, 1, 64, 10).astype(np.float32)
        labels = np.array(['a', 'a', 'b', 'b', 'c', 'c'])

        dataset = CSIDataset(
            data=data,
            labels=labels,
            sequence_length=10,
            preprocess=False,
            normalize=True,
        )

        assert dataset.n_classes == 3
        assert 'a' in dataset.label_to_idx
        assert 'b' in dataset.label_to_idx
        assert 'c' in dataset.label_to_idx

    def test_requires_data_source(self):
        """Test that dataset requires at least one data source."""
        with pytest.raises(ValueError):
            CSIDataset()
