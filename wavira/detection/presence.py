"""
Presence Detection Module

Detects human presence in a room using WiFi CSI (Channel State Information).
Presence causes changes in the wireless channel due to body reflections and
movement, which can be detected through variance analysis.

Features:
- Statistical variance analysis
- Motion detection via signal changes
- Configurable sensitivity thresholds
- Multi-subcarrier fusion
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class PresenceState:
    """Current presence detection state.

    Attributes:
        is_present: True if a person is detected
        is_moving: True if movement is detected
        presence_score: Presence likelihood score (0.0 to 1.0)
        motion_level: Relative motion level (0.0 to 1.0)
        variance_ratio: Current variance relative to baseline
    """
    is_present: bool
    is_moving: bool
    presence_score: float
    motion_level: float
    variance_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_present": self.is_present,
            "is_moving": self.is_moving,
            "presence_score": self.presence_score,
            "motion_level": self.motion_level,
            "variance_ratio": self.variance_ratio,
        }


@dataclass
class PresenceDetectorConfig:
    """Configuration for PresenceDetector.

    Attributes:
        sample_rate: Expected sample rate in Hz
        short_window_seconds: Duration of short-term analysis window
        long_window_seconds: Duration of long-term baseline window
        num_subcarriers: Number of CSI subcarriers
        presence_threshold: Variance ratio threshold for presence detection
        motion_threshold: Variance ratio threshold for motion detection
        min_samples: Minimum samples needed before detection
    """
    sample_rate: float = 10.0
    short_window_seconds: float = 2.0
    long_window_seconds: float = 30.0
    num_subcarriers: int = 52
    presence_threshold: float = 1.5
    motion_threshold: float = 3.0
    min_samples: int = 20


class PresenceDetector:
    """
    Presence detection using WiFi CSI data.

    This detector analyzes CSI amplitude variance to detect human presence
    and movement. It maintains both short-term and long-term statistics
    to adapt to environmental changes.

    The algorithm works by:
    1. Computing variance over a short window (recent activity)
    2. Comparing to variance over a long window (baseline)
    3. Detecting presence when short-term variance exceeds threshold

    Usage:
        detector = PresenceDetector()
        for amplitudes, timestamp in csi_stream:
            state = detector.update(amplitudes, timestamp)
            if state.is_present:
                print("Someone is in the room")
    """

    def __init__(self, config: Optional[PresenceDetectorConfig] = None):
        """
        Initialize presence detector.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or PresenceDetectorConfig()

        # Calculate window lengths
        self.short_window_length = int(
            self.config.short_window_seconds * self.config.sample_rate
        )
        self.long_window_length = int(
            self.config.long_window_seconds * self.config.sample_rate
        )

        # Per-subcarrier history buffers
        self.subcarrier_history: List[deque] = [
            deque(maxlen=self.long_window_length)
            for _ in range(self.config.num_subcarriers)
        ]
        self.time_history: deque = deque(maxlen=self.long_window_length)

        # Baseline statistics (from calibration or long-term average)
        self.baseline_variance: Optional[float] = None
        self.baseline_mean: Optional[float] = None

        # State tracking
        self._last_state: Optional[PresenceState] = None
        self._update_count = 0
        self._presence_count = 0

    def update(self, amplitudes: np.ndarray, timestamp: float) -> PresenceState:
        """
        Update detector with new CSI amplitude data.

        Args:
            amplitudes: Array of amplitude values for each subcarrier
            timestamp: Current timestamp in seconds

        Returns:
            PresenceState with detection results
        """
        self._update_count += 1

        # Handle amplitude array size
        amplitudes = np.asarray(amplitudes).flatten()
        if len(amplitudes) > self.config.num_subcarriers:
            amplitudes = amplitudes[:self.config.num_subcarriers]
        elif len(amplitudes) < self.config.num_subcarriers:
            amplitudes = np.pad(
                amplitudes,
                (0, self.config.num_subcarriers - len(amplitudes))
            )

        # Store in history
        for i, amp in enumerate(amplitudes):
            self.subcarrier_history[i].append(float(amp))
        self.time_history.append(timestamp)

        # Need minimum samples
        if len(self.time_history) < self.config.min_samples:
            return self._empty_state()

        # Compute short-term statistics
        short_stats = self._compute_stats(
            [list(h)[-self.short_window_length:] for h in self.subcarrier_history]
        )

        # Compute long-term statistics (for baseline if not calibrated)
        long_stats = self._compute_stats(
            [list(h) for h in self.subcarrier_history]
        )

        # Update baseline if not set
        if self.baseline_variance is None:
            self.baseline_variance = long_stats["variance"]
            self.baseline_mean = long_stats["mean"]

        # Calculate variance ratio
        baseline_var = max(self.baseline_variance, 1e-10)
        variance_ratio = short_stats["variance"] / baseline_var

        # Presence detection
        is_present = variance_ratio > self.config.presence_threshold

        # Motion detection (higher threshold)
        is_moving = variance_ratio > self.config.motion_threshold

        # Calculate scores
        presence_score = min(1.0, (variance_ratio - 1.0) /
                            (self.config.presence_threshold - 1.0 + 0.1))
        presence_score = max(0.0, presence_score)

        motion_level = min(1.0, (variance_ratio - self.config.presence_threshold) /
                          (self.config.motion_threshold - self.config.presence_threshold + 0.1))
        motion_level = max(0.0, motion_level)

        if is_present:
            self._presence_count += 1

        state = PresenceState(
            is_present=is_present,
            is_moving=is_moving,
            presence_score=presence_score,
            motion_level=motion_level,
            variance_ratio=variance_ratio,
        )
        self._last_state = state

        return state

    def _compute_stats(self, subcarrier_data: List[List[float]]) -> Dict[str, float]:
        """Compute statistics across all subcarriers."""
        all_data = []
        for data in subcarrier_data:
            if len(data) > 0:
                all_data.extend(data)

        if not all_data:
            return {"variance": 0.0, "mean": 0.0, "std": 0.0}

        arr = np.array(all_data)
        return {
            "variance": float(np.var(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def _empty_state(self) -> PresenceState:
        """Return empty state when not enough data."""
        return PresenceState(
            is_present=False,
            is_moving=False,
            presence_score=0.0,
            motion_level=0.0,
            variance_ratio=1.0,
        )

    def reset(self):
        """Reset detector state completely."""
        for hist in self.subcarrier_history:
            hist.clear()
        self.time_history.clear()
        self.baseline_variance = None
        self.baseline_mean = None
        self._last_state = None
        self._update_count = 0
        self._presence_count = 0

    def calibrate(self, amplitudes_list: List[np.ndarray], timestamps: List[float]):
        """
        Calibrate baseline from empty room data.

        Call this method with CSI data from an empty room to establish
        the baseline statistics for presence detection.

        Args:
            amplitudes_list: List of amplitude arrays
            timestamps: Corresponding timestamps
        """
        self.reset()

        for amplitudes, ts in zip(amplitudes_list, timestamps):
            self.update(amplitudes, ts)

        # Set baseline from current long-term statistics
        if len(self.time_history) >= self.config.min_samples:
            stats = self._compute_stats([list(h) for h in self.subcarrier_history])
            self.baseline_variance = stats["variance"]
            self.baseline_mean = stats["mean"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "update_count": self._update_count,
            "presence_count": self._presence_count,
            "presence_rate": self._presence_count / max(1, self._update_count),
            "samples_buffered": len(self.time_history),
            "baseline_variance": self.baseline_variance,
            "baseline_mean": self.baseline_mean,
            "last_state": self._last_state.to_dict() if self._last_state else None,
        }

    def set_sensitivity(self, level: str):
        """
        Set detection sensitivity.

        Args:
            level: One of 'low', 'medium', 'high'
        """
        if level == "low":
            self.config.presence_threshold = 2.0
            self.config.motion_threshold = 4.0
        elif level == "medium":
            self.config.presence_threshold = 1.5
            self.config.motion_threshold = 3.0
        elif level == "high":
            self.config.presence_threshold = 1.2
            self.config.motion_threshold = 2.0
        else:
            raise ValueError(f"Unknown sensitivity level: {level}")
