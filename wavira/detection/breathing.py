"""
Breathing Detection Module

Detects human breathing patterns from WiFi CSI (Channel State Information) signals.
Breathing causes subtle changes in the wireless channel due to chest movement,
which can be detected through bandpass filtering and spectral analysis.

Features:
- Multi-subcarrier analysis with weighted fusion
- Adaptive bandpass filtering (0.1-0.5 Hz for breathing)
- Kalman filter for noise reduction
- Presence detection based on signal variance
"""

import numpy as np
from scipy import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class BreathingState:
    """Current breathing detection state.

    Attributes:
        is_breathing: True if breathing is detected
        is_present: True if a person is detected in the room
        breath_rate: Estimated breathing rate in breaths per minute
        confidence: Detection confidence score (0.0 to 1.0)
        breath_ratio: Ratio of breathing band power to total power
        hold_duration: Duration in seconds if breath is being held
    """
    is_breathing: bool
    is_present: bool
    breath_rate: float  # breaths per minute
    confidence: float  # 0-1
    breath_ratio: float  # breathing band power ratio
    hold_duration: float  # seconds if holding breath

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_breathing": self.is_breathing,
            "is_present": self.is_present,
            "breath_rate": self.breath_rate,
            "confidence": self.confidence,
            "breath_ratio": self.breath_ratio,
            "hold_duration": self.hold_duration,
        }


@dataclass
class BreathingDetectorConfig:
    """Configuration for BreathingDetector.

    Attributes:
        sample_rate: Expected sample rate in Hz
        history_seconds: Duration of history to maintain
        num_subcarriers: Number of CSI subcarriers
        breathing_freq_min: Minimum breathing frequency in Hz (6 breaths/min)
        breathing_freq_max: Maximum breathing frequency in Hz (30 breaths/min)
        min_samples: Minimum samples needed before detection
        breathing_threshold: Threshold for breathing detection
        presence_threshold: Variance multiplier for presence detection
        kalman_process_var: Kalman filter process variance
        kalman_measure_var: Kalman filter measurement variance
    """
    sample_rate: float = 10.0
    history_seconds: float = 15.0
    num_subcarriers: int = 52
    breathing_freq_min: float = 0.1  # 6 breaths/min
    breathing_freq_max: float = 0.5  # 30 breaths/min
    min_samples: int = 30
    breathing_threshold: float = 0.08
    presence_threshold: float = 0.5
    kalman_process_var: float = 1e-3
    kalman_measure_var: float = 1e-1


class KalmanFilter1D:
    """Simple 1D Kalman filter for signal smoothing.

    Implements a basic Kalman filter for scalar measurements with
    constant velocity model. Useful for reducing noise in CSI signals.

    Args:
        process_variance: Process noise variance (Q)
        measurement_variance: Measurement noise variance (R)
    """

    def __init__(
        self,
        process_variance: float = 1e-3,
        measurement_variance: float = 1e-1
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_estimate = 1.0

    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement.

        Args:
            measurement: New measurement value

        Returns:
            Filtered estimate
        """
        # Prediction step
        prediction = self.estimate
        error_prediction = self.error_estimate + self.process_variance

        # Update step
        kalman_gain = error_prediction / (error_prediction + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * error_prediction

        return self.estimate

    def reset(self):
        """Reset filter state."""
        self.estimate = 0.0
        self.error_estimate = 1.0

    def get_state(self) -> Dict[str, float]:
        """Get current filter state."""
        return {
            "estimate": self.estimate,
            "error_estimate": self.error_estimate,
        }


class BreathingDetector:
    """
    Advanced breathing detection using WiFi CSI data.

    This detector analyzes CSI amplitude patterns to detect human breathing.
    It uses multiple techniques for robust detection:

    1. Multi-subcarrier analysis: Analyzes all subcarriers and fuses results
    2. Bandpass filtering: Isolates the breathing frequency band (0.1-0.5 Hz)
    3. Kalman filtering: Reduces noise in individual subcarrier signals
    4. Variance-based presence detection: Determines if someone is present

    Usage:
        detector = BreathingDetector()
        for amplitudes, timestamp in csi_stream:
            state = detector.update(amplitudes, timestamp)
            if state.is_breathing:
                print(f"Breathing detected: {state.breath_rate:.1f} breaths/min")
    """

    def __init__(self, config: Optional[BreathingDetectorConfig] = None):
        """
        Initialize breathing detector.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or BreathingDetectorConfig()

        # Calculate history length
        self.history_length = int(self.config.history_seconds * self.config.sample_rate)

        # Per-subcarrier history buffers
        self.subcarrier_history: List[deque] = [
            deque(maxlen=self.history_length)
            for _ in range(self.config.num_subcarriers)
        ]
        self.time_history: deque = deque(maxlen=self.history_length)

        # Kalman filters for each subcarrier
        self.kalman_filters = [
            KalmanFilter1D(
                self.config.kalman_process_var,
                self.config.kalman_measure_var
            )
            for _ in range(self.config.num_subcarriers)
        ]

        # State tracking
        self.breath_hold_start: Optional[float] = None
        self.baseline_variance: Optional[float] = None

        # Subcarrier weights (learned over time)
        self.subcarrier_weights = np.ones(self.config.num_subcarriers) / self.config.num_subcarriers

        # Statistics
        self._update_count = 0
        self._detection_count = 0

    def update(self, amplitudes: np.ndarray, timestamp: float) -> BreathingState:
        """
        Update detector with new CSI amplitude data.

        This method should be called for each new CSI sample. It processes
        the amplitudes, updates internal state, and returns detection results.

        Args:
            amplitudes: Array of amplitude values for each subcarrier.
                       Shape should be (num_subcarriers,) or larger.
            timestamp: Current timestamp in seconds

        Returns:
            BreathingState with detection results
        """
        self._update_count += 1

        # Handle amplitude array size mismatch
        amplitudes = np.asarray(amplitudes).flatten()
        if len(amplitudes) > self.config.num_subcarriers:
            amplitudes = amplitudes[:self.config.num_subcarriers]
        elif len(amplitudes) < self.config.num_subcarriers:
            amplitudes = np.pad(
                amplitudes,
                (0, self.config.num_subcarriers - len(amplitudes))
            )

        # Apply Kalman filter and store filtered values
        for i, (amp, kf) in enumerate(zip(amplitudes, self.kalman_filters)):
            filtered = kf.update(float(amp))
            self.subcarrier_history[i].append(filtered)

        self.time_history.append(timestamp)

        # Need enough samples for analysis
        if len(self.time_history) < self.config.min_samples:
            return self._empty_state()

        # Calculate actual sample rate from timestamps
        times = np.array(self.time_history)
        duration = times[-1] - times[0]
        actual_sample_rate = len(times) / duration if duration > 0 else self.config.sample_rate

        # Analyze each subcarrier for breathing signal
        breathing_ratios = []
        peak_frequencies = []

        for i, hist in enumerate(self.subcarrier_history):
            if len(hist) < self.config.min_samples:
                continue

            ratio, peak_freq = self._analyze_subcarrier(
                np.array(list(hist)),
                actual_sample_rate
            )

            if ratio is not None:
                breathing_ratios.append(ratio)
                if peak_freq is not None:
                    peak_frequencies.append(peak_freq)

        if not breathing_ratios:
            return self._empty_state()

        # Aggregate results from all subcarriers
        breathing_ratios = np.array(breathing_ratios)
        best_ratio = float(np.max(breathing_ratios))

        # Update subcarrier weights based on signal quality
        self._update_weights(breathing_ratios)

        # Presence detection based on variance
        total_variance = np.mean([
            np.var(list(h))
            for h in self.subcarrier_history
            if len(h) > 10
        ])

        if self.baseline_variance is None:
            self.baseline_variance = float(total_variance)

        is_present = bool(total_variance > self.baseline_variance * self.config.presence_threshold)

        # Breathing detection
        is_breathing = bool(best_ratio > self.config.breathing_threshold and is_present)

        if is_breathing:
            self._detection_count += 1

        # Calculate breath rate from peak frequencies
        if peak_frequencies:
            breath_rate = float(np.median(peak_frequencies)) * 60  # Convert to breaths/min
        else:
            breath_rate = 0.0

        # Calculate confidence
        confidence = min(1.0, best_ratio * 2) if is_breathing else 0.0

        # Track breath holding
        hold_duration = self._update_breath_hold(is_breathing, is_present, timestamp)

        return BreathingState(
            is_breathing=is_breathing,
            is_present=is_present,
            breath_rate=breath_rate,
            confidence=confidence,
            breath_ratio=best_ratio,
            hold_duration=hold_duration,
        )

    def _analyze_subcarrier(
        self,
        samples: np.ndarray,
        sample_rate: float
    ) -> tuple:
        """
        Analyze a single subcarrier for breathing signal.

        Returns:
            Tuple of (breathing_ratio, peak_frequency) or (None, None) on error
        """
        try:
            # Detrend (remove DC component)
            samples = samples - np.mean(samples)

            # Design and apply bandpass filter for breathing frequencies
            sos = signal.butter(
                2,
                [self.config.breathing_freq_min, self.config.breathing_freq_max],
                btype='band',
                fs=sample_rate,
                output='sos'
            )
            filtered = signal.sosfilt(sos, samples)

            # Calculate breathing band power ratio
            breath_var = np.var(filtered)
            total_var = np.var(samples) + 1e-10
            ratio = breath_var / total_var

            peak_freq = None

            # Find peak frequency if significant breathing signal
            if ratio > 0.05:
                n = len(samples)
                # Apply Hanning window before FFT
                windowed = samples * np.hanning(n)
                fft_vals = np.abs(np.fft.rfft(windowed))
                freqs = np.fft.rfftfreq(n, 1 / sample_rate)

                # Find peak in breathing frequency band
                breath_mask = (
                    (freqs >= self.config.breathing_freq_min) &
                    (freqs <= self.config.breathing_freq_max)
                )
                if np.any(breath_mask):
                    breath_fft = fft_vals[breath_mask]
                    breath_freqs = freqs[breath_mask]
                    peak_idx = np.argmax(breath_fft)
                    peak_freq = float(breath_freqs[peak_idx])

            return ratio, peak_freq

        except Exception:
            return None, None

    def _update_weights(self, ratios: np.ndarray):
        """Update subcarrier weights based on breathing signal quality."""
        if len(ratios) < self.config.num_subcarriers:
            return

        # Normalize ratios to weights
        total = np.sum(ratios) + 1e-10
        weights = ratios / total

        # Exponential moving average update
        alpha = 0.1
        self.subcarrier_weights[:len(weights)] = (
            alpha * weights +
            (1 - alpha) * self.subcarrier_weights[:len(weights)]
        )

    def _update_breath_hold(
        self,
        is_breathing: bool,
        is_present: bool,
        timestamp: float
    ) -> float:
        """Track breath holding duration."""
        if not is_breathing and is_present:
            if self.breath_hold_start is None:
                self.breath_hold_start = timestamp
            return timestamp - self.breath_hold_start
        else:
            self.breath_hold_start = None
            return 0.0

    def _empty_state(self) -> BreathingState:
        """Return empty state when not enough data."""
        return BreathingState(
            is_breathing=False,
            is_present=False,
            breath_rate=0.0,
            confidence=0.0,
            breath_ratio=0.0,
            hold_duration=0.0,
        )

    def reset(self):
        """Reset detector state completely."""
        for hist in self.subcarrier_history:
            hist.clear()
        self.time_history.clear()
        for kf in self.kalman_filters:
            kf.reset()
        self.breath_hold_start = None
        self.baseline_variance = None
        self.subcarrier_weights = np.ones(self.config.num_subcarriers) / self.config.num_subcarriers
        self._update_count = 0
        self._detection_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "update_count": self._update_count,
            "detection_count": self._detection_count,
            "detection_rate": self._detection_count / max(1, self._update_count),
            "samples_buffered": len(self.time_history),
            "baseline_variance": self.baseline_variance,
        }

    def calibrate_baseline(self, amplitudes_list: List[np.ndarray], timestamps: List[float]):
        """
        Calibrate baseline variance from empty room data.

        Call this method with CSI data from an empty room to establish
        the baseline variance for presence detection.

        Args:
            amplitudes_list: List of amplitude arrays
            timestamps: Corresponding timestamps
        """
        for amplitudes, ts in zip(amplitudes_list, timestamps):
            self.update(amplitudes, ts)

        # Set baseline from current variance
        if len(self.time_history) >= self.config.min_samples:
            self.baseline_variance = np.mean([
                np.var(list(h))
                for h in self.subcarrier_history
                if len(h) > 10
            ])
