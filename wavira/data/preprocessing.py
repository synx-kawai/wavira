"""
CSI Signal Preprocessing Module

Implements preprocessing steps for Wi-Fi Channel State Information:
- Amplitude extraction from complex CSI values
- Hampel filtering for outlier removal
- Phase sanitization to remove linear phase shifts
"""

import numpy as np
from typing import Tuple


def extract_amplitude(csi_complex: np.ndarray) -> np.ndarray:
    """
    Extract amplitude from complex CSI values.

    Args:
        csi_complex: Complex CSI matrix of shape (rx_antennas, subcarriers, packets)
                     or (rx_antennas, tx_antennas, subcarriers, packets)

    Returns:
        Amplitude values with same shape as input
    """
    return np.abs(csi_complex)


def extract_phase(csi_complex: np.ndarray) -> np.ndarray:
    """
    Extract phase from complex CSI values.

    Args:
        csi_complex: Complex CSI matrix

    Returns:
        Phase values in radians
    """
    return np.angle(csi_complex)


def hampel_filter(
    signal: np.ndarray,
    window_size: int = 5,
    threshold: float = 3.0,
    axis: int = -1
) -> np.ndarray:
    """
    Apply Hampel filter to remove outliers based on median absolute deviation.

    The Hampel filter identifies outliers by comparing each point to the median
    of its neighboring values. Points that deviate more than threshold * MAD
    from the median are replaced with the median.

    Args:
        signal: Input signal array
        window_size: Half-window size for median calculation (w in paper)
        threshold: Number of MADs to consider as outlier (xi in paper)
        axis: Axis along which to apply the filter

    Returns:
        Filtered signal with outliers replaced
    """
    signal = np.asarray(signal, dtype=np.float64)
    filtered = signal.copy()

    # Move target axis to the end for easier processing
    filtered = np.moveaxis(filtered, axis, -1)
    original_shape = filtered.shape
    n_samples = original_shape[-1]

    # Flatten all dimensions except the last
    flat_shape = (-1, n_samples)
    filtered = filtered.reshape(flat_shape)

    # Scale factor for MAD to standard deviation
    k = 1.4826

    for i in range(n_samples):
        # Define window boundaries
        start = max(0, i - window_size)
        end = min(n_samples, i + window_size + 1)

        # Get window values for all signals
        window = filtered[:, start:end]

        # Calculate median and MAD for each signal
        median = np.median(window, axis=1, keepdims=True)
        mad = k * np.median(np.abs(window - median), axis=1)

        # Identify outliers
        deviation = np.abs(filtered[:, i] - median.squeeze())
        outlier_mask = deviation > threshold * mad

        # Replace outliers with median
        filtered[outlier_mask, i] = median.squeeze()[outlier_mask]

    # Restore original shape
    filtered = filtered.reshape(original_shape)
    filtered = np.moveaxis(filtered, -1, axis)

    return filtered


def phase_sanitization(phase: np.ndarray, subcarrier_indices: np.ndarray = None) -> np.ndarray:
    """
    Remove linear phase shifts from CSI phase measurements.

    Phase errors in Wi-Fi CSI are caused by:
    - Sampling Frequency Offset (SFO)
    - Carrier Frequency Offset (CFO)
    - Packet Detection Delay (PDD)

    This function estimates and removes the linear component of phase
    across subcarriers using least-squares fitting.

    Args:
        phase: Phase values of shape (..., subcarriers, packets)
        subcarrier_indices: Subcarrier index values. If None, uses 0 to N-1.

    Returns:
        Sanitized phase with linear component removed
    """
    phase = np.asarray(phase, dtype=np.float64)

    # Assume subcarriers are second-to-last dimension
    n_subcarriers = phase.shape[-2]

    if subcarrier_indices is None:
        subcarrier_indices = np.arange(n_subcarriers)

    subcarrier_indices = np.asarray(subcarrier_indices, dtype=np.float64)

    # Reshape for processing: flatten all but last two dimensions
    original_shape = phase.shape
    phase = phase.reshape(-1, n_subcarriers, original_shape[-1])
    sanitized = np.zeros_like(phase)

    # For each sample (antenna combination and packet)
    for batch_idx in range(phase.shape[0]):
        for packet_idx in range(phase.shape[-1]):
            phi = phase[batch_idx, :, packet_idx]

            # Unwrap phase to handle discontinuities
            phi_unwrapped = np.unwrap(phi)

            # Least-squares fit: phi = a * k + b
            # where k is subcarrier index
            A = np.vstack([subcarrier_indices, np.ones(n_subcarriers)]).T
            slope, offset = np.linalg.lstsq(A, phi_unwrapped, rcond=None)[0]

            # Remove linear component
            linear_component = slope * subcarrier_indices + offset
            sanitized[batch_idx, :, packet_idx] = phi_unwrapped - linear_component

    # Restore original shape
    sanitized = sanitized.reshape(original_shape)

    return sanitized


def preprocess_csi(
    csi_complex: np.ndarray,
    hampel_window: int = 5,
    hampel_threshold: float = 3.0,
    use_phase: bool = False,
    subcarrier_indices: np.ndarray = None
) -> np.ndarray:
    """
    Full preprocessing pipeline for CSI data.

    Applies the following steps:
    1. Extract amplitude (and optionally phase)
    2. Apply Hampel filtering to remove outliers
    3. If using phase, apply phase sanitization

    Args:
        csi_complex: Complex CSI matrix of shape (rx_antennas, subcarriers, packets)
        hampel_window: Window size for Hampel filter
        hampel_threshold: Threshold for Hampel filter
        use_phase: Whether to include sanitized phase information
        subcarrier_indices: Subcarrier indices for phase sanitization

    Returns:
        Preprocessed CSI data. If use_phase=False, returns amplitude only.
        If use_phase=True, returns concatenated amplitude and sanitized phase.
    """
    # Extract amplitude
    amplitude = extract_amplitude(csi_complex)

    # Apply Hampel filter along packet dimension
    amplitude = hampel_filter(
        amplitude,
        window_size=hampel_window,
        threshold=hampel_threshold,
        axis=-1
    )

    if not use_phase:
        return amplitude.astype(np.float32)

    # Extract and sanitize phase
    phase = extract_phase(csi_complex)
    phase = phase_sanitization(phase, subcarrier_indices)

    # Apply Hampel filter to phase
    phase = hampel_filter(
        phase,
        window_size=hampel_window,
        threshold=hampel_threshold,
        axis=-1
    )

    # Concatenate amplitude and phase along a new dimension
    combined = np.stack([amplitude, phase], axis=0)

    return combined.astype(np.float32)


def normalize_csi(csi: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize CSI data to zero mean and unit variance.

    Args:
        csi: CSI data array
        axis: Axis along which to compute statistics

    Returns:
        Tuple of (normalized_csi, mean, std)
    """
    mean = np.mean(csi, axis=axis, keepdims=True)
    std = np.std(csi, axis=axis, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    normalized = (csi - mean) / std

    return normalized.astype(np.float32), mean, std
