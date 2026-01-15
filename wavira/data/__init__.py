"""Data processing modules for CSI signals."""

from wavira.data.preprocessing import (
    hampel_filter,
    phase_sanitization,
    extract_amplitude,
    preprocess_csi,
)
from wavira.data.dataset import CSIDataset
from wavira.data.crowd_dataset import CrowdDataset, SyntheticCrowdDataset
from wavira.data.gesture_dataset import (
    GestureDataset,
    DualDeviceGestureDataset,
    SyntheticGestureDataset,
    create_gesture_dataloaders,
)
from wavira.data.gesture_preprocessing import (
    butterworth_lowpass_filter,
    butterworth_bandpass_filter,
    preprocess_gesture_csi,
    GesturePreprocessor,
    GesturePreprocessConfig,
    GestureAugmentor,
)

__all__ = [
    "hampel_filter",
    "phase_sanitization",
    "extract_amplitude",
    "preprocess_csi",
    "CSIDataset",
    "CrowdDataset",
    "SyntheticCrowdDataset",
    # Gesture recognition
    "GestureDataset",
    "DualDeviceGestureDataset",
    "SyntheticGestureDataset",
    "create_gesture_dataloaders",
    "butterworth_lowpass_filter",
    "butterworth_bandpass_filter",
    "preprocess_gesture_csi",
    "GesturePreprocessor",
    "GesturePreprocessConfig",
    "GestureAugmentor",
]
