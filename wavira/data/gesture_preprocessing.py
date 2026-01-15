"""
Gesture Recognition Preprocessing Module

Preprocessing pipeline for CSI data used in gesture recognition.
Implements Butterworth lowpass filtering as recommended in the paper.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class GesturePreprocessConfig:
    """Configuration for gesture preprocessing pipeline."""
    # Butterworth filter parameters
    butter_order: int = 4  # Filter order
    butter_cutoff: float = 20.0  # Cutoff frequency in Hz
    sampling_rate: float = 100.0  # CSI sampling rate (100 Hz = 0.01 sec interval)

    # Normalization parameters
    normalize: bool = True
    normalize_per_subcarrier: bool = True

    # Amplitude extraction
    extract_amplitude: bool = True


def butterworth_lowpass_filter(
    data: np.ndarray,
    cutoff: float = 20.0,
    sampling_rate: float = 100.0,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Apply Butterworth lowpass filter for noise removal.

    The paper recommends using a lowpass Butterworth filter to remove
    high-frequency noise from CSI amplitude data.

    Args:
        data: Input signal array
        cutoff: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order
        axis: Axis along which to apply filter

    Returns:
        Filtered signal
    """
    # Normalize cutoff frequency
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff / nyquist

    # Ensure cutoff is valid
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99

    # Design Butterworth filter
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply filter using filtfilt for zero phase distortion
    filtered = signal.filtfilt(b, a, data, axis=axis)

    return filtered


def butterworth_bandpass_filter(
    data: np.ndarray,
    low_cutoff: float = 0.5,
    high_cutoff: float = 20.0,
    sampling_rate: float = 100.0,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Useful for isolating gesture-related frequency components
    while removing both DC offset and high-frequency noise.

    Args:
        data: Input signal array
        low_cutoff: Lower cutoff frequency in Hz
        high_cutoff: Upper cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order
        axis: Axis along which to apply filter

    Returns:
        Filtered signal
    """
    nyquist = sampling_rate / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    # Ensure valid range
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    filtered = signal.filtfilt(b, a, data, axis=axis)

    return filtered


def extract_amplitude(csi_complex: np.ndarray) -> np.ndarray:
    """
    Extract amplitude from complex CSI values.

    Args:
        csi_complex: Complex CSI matrix

    Returns:
        Amplitude values
    """
    return np.abs(csi_complex)


def normalize_csi(
    csi: np.ndarray,
    per_subcarrier: bool = True,
    axis: int = -1
) -> np.ndarray:
    """
    Normalize CSI data to zero mean and unit variance.

    Args:
        csi: CSI data array of shape (..., n_subcarriers, n_frames)
        per_subcarrier: Normalize each subcarrier independently
        axis: Time axis for computing statistics

    Returns:
        Normalized CSI data
    """
    if per_subcarrier:
        # Normalize each subcarrier's time series independently
        mean = np.mean(csi, axis=axis, keepdims=True)
        std = np.std(csi, axis=axis, keepdims=True)
    else:
        # Global normalization
        mean = np.mean(csi)
        std = np.std(csi)

    std = np.where(std == 0, 1, std)
    normalized = (csi - mean) / std

    return normalized.astype(np.float32)


def preprocess_gesture_csi(
    csi_data: np.ndarray,
    config: Optional[GesturePreprocessConfig] = None,
) -> np.ndarray:
    """
    Full preprocessing pipeline for gesture recognition CSI data.

    Pipeline:
    1. Extract amplitude (if complex input)
    2. Apply Butterworth lowpass filter
    3. Normalize data

    Args:
        csi_data: Raw CSI data of shape (n_routes, n_subcarriers, n_frames)
        config: Preprocessing configuration

    Returns:
        Preprocessed CSI data
    """
    if config is None:
        config = GesturePreprocessConfig()

    # Extract amplitude if data is complex
    if config.extract_amplitude and np.iscomplexobj(csi_data):
        csi_data = extract_amplitude(csi_data)

    # Ensure float type
    csi_data = csi_data.astype(np.float64)

    # Apply Butterworth lowpass filter along time axis
    csi_data = butterworth_lowpass_filter(
        csi_data,
        cutoff=config.butter_cutoff,
        sampling_rate=config.sampling_rate,
        order=config.butter_order,
        axis=-1  # Time is last dimension
    )

    # Normalize
    if config.normalize:
        csi_data = normalize_csi(
            csi_data,
            per_subcarrier=config.normalize_per_subcarrier,
            axis=-1
        )

    return csi_data.astype(np.float32)


class GesturePreprocessor:
    """
    Callable preprocessor class for use with datasets.

    Example:
        preprocessor = GesturePreprocessor(cutoff=15.0)
        dataset = GestureDataset(preprocess_fn=preprocessor)
    """

    def __init__(
        self,
        butter_order: int = 4,
        butter_cutoff: float = 20.0,
        sampling_rate: float = 100.0,
        normalize: bool = True,
        normalize_per_subcarrier: bool = True,
    ):
        self.config = GesturePreprocessConfig(
            butter_order=butter_order,
            butter_cutoff=butter_cutoff,
            sampling_rate=sampling_rate,
            normalize=normalize,
            normalize_per_subcarrier=normalize_per_subcarrier,
        )

    def __call__(self, csi_data: np.ndarray) -> np.ndarray:
        return preprocess_gesture_csi(csi_data, self.config)


def segment_gesture(
    csi_data: np.ndarray,
    threshold: float = 0.3,
    min_length: int = 20,
    max_length: int = 200,
) -> Tuple[int, int]:
    """
    Automatically segment gesture from continuous CSI stream.

    Uses variance-based detection to find gesture boundaries.

    Args:
        csi_data: CSI data of shape (n_routes, n_subcarriers, n_frames)
        threshold: Variance threshold for gesture detection
        min_length: Minimum gesture length in frames
        max_length: Maximum gesture length in frames

    Returns:
        Tuple of (start_frame, end_frame)
    """
    # Compute variance across routes and subcarriers for each frame
    variance = np.var(csi_data, axis=(0, 1))

    # Normalize variance
    variance_norm = variance / (variance.max() + 1e-8)

    # Find frames above threshold
    active_frames = variance_norm > threshold

    # Find start and end
    active_indices = np.where(active_frames)[0]

    if len(active_indices) < min_length:
        # No clear gesture detected, return full range
        return 0, csi_data.shape[-1]

    start = max(0, active_indices[0] - 5)  # Add small padding
    end = min(csi_data.shape[-1], active_indices[-1] + 5)

    # Enforce length constraints
    length = end - start
    if length < min_length:
        # Extend to minimum length
        padding = (min_length - length) // 2
        start = max(0, start - padding)
        end = min(csi_data.shape[-1], end + padding)
    elif length > max_length:
        # Truncate to maximum length
        center = (start + end) // 2
        start = center - max_length // 2
        end = center + max_length // 2

    return int(start), int(end)


def augment_gesture_csi(
    csi_data: np.ndarray,
    time_shift_range: Tuple[int, int] = (-5, 5),
    noise_level: float = 0.05,
    scale_range: Tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """
    Apply data augmentation to gesture CSI data.

    Args:
        csi_data: CSI data of shape (n_routes, n_subcarriers, n_frames)
        time_shift_range: Range for random time shift
        noise_level: Gaussian noise level (fraction of signal std)
        scale_range: Range for random amplitude scaling

    Returns:
        Augmented CSI data
    """
    augmented = csi_data.copy()

    # Random time shift
    shift = np.random.randint(time_shift_range[0], time_shift_range[1] + 1)
    if shift != 0:
        augmented = np.roll(augmented, shift, axis=-1)
        # Zero pad shifted region
        if shift > 0:
            augmented[..., :shift] = augmented[..., shift:shift+1]
        else:
            augmented[..., shift:] = augmented[..., shift-1:shift]

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * np.std(augmented), augmented.shape)
    augmented += noise

    # Random amplitude scaling
    scale = np.random.uniform(scale_range[0], scale_range[1])
    augmented *= scale

    return augmented.astype(np.float32)


class GestureAugmentor:
    """
    Callable augmentation class for training data augmentation.
    """

    def __init__(
        self,
        time_shift_range: Tuple[int, int] = (-5, 5),
        noise_level: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,  # Probability of applying augmentation
    ):
        self.time_shift_range = time_shift_range
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.p = p

    def __call__(self, csi_data: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return augment_gesture_csi(
                csi_data,
                self.time_shift_range,
                self.noise_level,
                self.scale_range,
            )
        return csi_data
