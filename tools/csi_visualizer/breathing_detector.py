#!/usr/bin/env python3
"""
Improved Breathing Detection Algorithm
Issue #7: 呼吸・存在検知アルゴリズムの改善

Features:
- Multi-subcarrier analysis with weighted fusion
- Adaptive bandpass filtering (0.1-0.5 Hz for breathing)
- Kalman filter for noise reduction
- Presence detection based on signal variance
"""

import numpy as np
from scipy import signal
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class BreathingState:
    """Current breathing detection state."""
    is_breathing: bool
    is_present: bool
    breath_rate: float  # breaths per minute
    confidence: float  # 0-1
    breath_ratio: float  # breathing band power ratio
    hold_duration: float  # seconds if holding breath


class KalmanFilter1D:
    """Simple 1D Kalman filter for signal smoothing."""

    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_estimate = 1.0

    def update(self, measurement: float) -> float:
        # Prediction
        prediction = self.estimate
        error_prediction = self.error_estimate + self.process_variance

        # Update
        kalman_gain = error_prediction / (error_prediction + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * error_prediction

        return self.estimate

    def reset(self):
        self.estimate = 0.0
        self.error_estimate = 1.0


class BreathingDetector:
    """
    Advanced breathing detection using WiFi CSI data.

    Improvements over basic FFT approach:
    1. Uses multiple subcarriers with weighted fusion
    2. Adaptive bandpass filtering
    3. Kalman filtering for noise reduction
    4. Presence detection based on variance
    """

    def __init__(
        self,
        sample_rate: float = 10.0,
        history_seconds: float = 15.0,
        num_subcarriers: int = 52,
        breathing_freq_min: float = 0.1,  # 6 breaths/min
        breathing_freq_max: float = 0.5,  # 30 breaths/min
    ):
        self.sample_rate = sample_rate
        self.history_length = int(history_seconds * sample_rate)
        self.num_subcarriers = num_subcarriers
        self.breathing_freq_min = breathing_freq_min
        self.breathing_freq_max = breathing_freq_max

        # Per-subcarrier history
        self.subcarrier_history: List[deque] = [
            deque(maxlen=self.history_length) for _ in range(num_subcarriers)
        ]
        self.time_history = deque(maxlen=self.history_length)

        # Kalman filters for each subcarrier
        self.kalman_filters = [KalmanFilter1D() for _ in range(num_subcarriers)]

        # State tracking
        self.breath_hold_start: Optional[float] = None
        self.baseline_variance: Optional[float] = None
        self.presence_threshold = 2.0  # Variance multiplier for presence

        # Subcarrier weights (learned over time)
        self.subcarrier_weights = np.ones(num_subcarriers) / num_subcarriers

    def update(self, amplitudes: np.ndarray, timestamp: float) -> BreathingState:
        """
        Update with new CSI amplitude data.

        Args:
            amplitudes: Array of amplitude values for each subcarrier
            timestamp: Current timestamp

        Returns:
            BreathingState with detection results
        """
        # Ensure correct size
        if len(amplitudes) > self.num_subcarriers:
            amplitudes = amplitudes[:self.num_subcarriers]
        elif len(amplitudes) < self.num_subcarriers:
            amplitudes = np.pad(amplitudes, (0, self.num_subcarriers - len(amplitudes)))

        # Apply Kalman filter and store
        for i, (amp, kf) in enumerate(zip(amplitudes, self.kalman_filters)):
            filtered = kf.update(amp)
            self.subcarrier_history[i].append(filtered)

        self.time_history.append(timestamp)

        # Need enough samples for analysis
        if len(self.time_history) < 30:
            return BreathingState(
                is_breathing=False,
                is_present=False,
                breath_rate=0,
                confidence=0,
                breath_ratio=0,
                hold_duration=0
            )

        # Calculate actual sample rate
        times = np.array(self.time_history)
        duration = times[-1] - times[0]
        actual_sample_rate = len(times) / duration if duration > 0 else self.sample_rate

        # Analyze each subcarrier
        breathing_ratios = []
        peak_frequencies = []

        for i, hist in enumerate(self.subcarrier_history):
            if len(hist) < 30:
                continue

            samples = np.array(list(hist))

            # Detrend
            samples = samples - np.mean(samples)

            # Apply bandpass filter
            try:
                sos = signal.butter(
                    2,
                    [self.breathing_freq_min, self.breathing_freq_max],
                    btype='band',
                    fs=actual_sample_rate,
                    output='sos'
                )
                filtered = signal.sosfilt(sos, samples)

                # Calculate breathing band power ratio
                breath_var = np.var(filtered)
                total_var = np.var(samples) + 1e-10
                ratio = breath_var / total_var
                breathing_ratios.append(ratio)

                # Find peak frequency using FFT
                if ratio > 0.05:  # Only if significant breathing signal
                    n = len(samples)
                    fft_vals = np.abs(np.fft.rfft(samples * np.hanning(n)))
                    freqs = np.fft.rfftfreq(n, 1 / actual_sample_rate)

                    breath_mask = (freqs >= self.breathing_freq_min) & (freqs <= self.breathing_freq_max)
                    if np.any(breath_mask):
                        breath_fft = fft_vals[breath_mask]
                        breath_freqs = freqs[breath_mask]
                        peak_idx = np.argmax(breath_fft)
                        peak_frequencies.append(breath_freqs[peak_idx])

            except Exception:
                continue

        if not breathing_ratios:
            return BreathingState(
                is_breathing=False,
                is_present=False,
                breath_rate=0,
                confidence=0,
                breath_ratio=0,
                hold_duration=0
            )

        # Weighted fusion of subcarrier results
        breathing_ratios = np.array(breathing_ratios)
        best_ratio = np.max(breathing_ratios)
        median_ratio = np.median(breathing_ratios)

        # Update subcarrier weights based on performance
        self._update_weights(breathing_ratios)

        # Presence detection based on variance
        total_variance = np.mean([np.var(list(h)) for h in self.subcarrier_history if len(h) > 10])

        if self.baseline_variance is None:
            self.baseline_variance = total_variance

        is_present = total_variance > self.baseline_variance * 0.5

        # Breathing detection
        breathing_threshold = 0.08
        is_breathing = best_ratio > breathing_threshold and is_present

        # Calculate breath rate
        if peak_frequencies:
            breath_rate = np.median(peak_frequencies) * 60  # Convert to breaths/min
        else:
            breath_rate = 0

        # Confidence based on consistency
        confidence = min(1.0, best_ratio * 2) if is_breathing else 0.0

        # Track breath holding
        now = timestamp
        if not is_breathing and is_present:
            if self.breath_hold_start is None:
                self.breath_hold_start = now
            hold_duration = now - self.breath_hold_start
        else:
            self.breath_hold_start = None
            hold_duration = 0

        return BreathingState(
            is_breathing=is_breathing,
            is_present=is_present,
            breath_rate=breath_rate,
            confidence=confidence,
            breath_ratio=best_ratio,
            hold_duration=hold_duration
        )

    def _update_weights(self, ratios: np.ndarray):
        """Update subcarrier weights based on breathing signal quality."""
        if len(ratios) < self.num_subcarriers:
            return

        # Normalize ratios to weights
        weights = ratios / (np.sum(ratios) + 1e-10)

        # Exponential moving average update
        alpha = 0.1
        self.subcarrier_weights[:len(weights)] = (
            alpha * weights + (1 - alpha) * self.subcarrier_weights[:len(weights)]
        )

    def reset(self):
        """Reset detector state."""
        for hist in self.subcarrier_history:
            hist.clear()
        self.time_history.clear()
        for kf in self.kalman_filters:
            kf.reset()
        self.breath_hold_start = None
        self.baseline_variance = None


# Test function
def test_breathing_detector():
    """Test the breathing detector with synthetic data."""
    import time

    detector = BreathingDetector(sample_rate=10.0)

    print("Testing Breathing Detector with synthetic data...")
    print("-" * 50)

    # Simulate 20 seconds of data
    for i in range(200):
        t = i * 0.1  # 10 Hz

        # Simulate breathing at 0.25 Hz (15 breaths/min) + noise
        breathing_signal = 10 * np.sin(2 * np.pi * 0.25 * t)
        noise = np.random.normal(0, 2, 52)
        amplitudes = 50 + breathing_signal + noise

        state = detector.update(amplitudes, t)

        if i % 20 == 0 and i > 30:
            print(f"[{t:.1f}s] breathing={state.is_breathing}, "
                  f"rate={state.breath_rate:.1f}/min, "
                  f"ratio={state.breath_ratio:.3f}, "
                  f"conf={state.confidence:.2f}")

    print("-" * 50)
    print("Test complete!")


if __name__ == '__main__':
    test_breathing_detector()
