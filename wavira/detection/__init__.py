"""Detection modules for presence and breathing detection."""

from wavira.detection.breathing import (
    BreathingState,
    KalmanFilter1D,
    BreathingDetector,
    BreathingDetectorConfig,
)
from wavira.detection.presence import (
    PresenceState,
    PresenceDetector,
    PresenceDetectorConfig,
)

__all__ = [
    # Breathing detection
    "BreathingState",
    "KalmanFilter1D",
    "BreathingDetector",
    "BreathingDetectorConfig",
    # Presence detection
    "PresenceState",
    "PresenceDetector",
    "PresenceDetectorConfig",
]
