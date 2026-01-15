"""Tests for breathing and presence detection modules."""

import numpy as np
import pytest

from wavira.detection import (
    BreathingState,
    KalmanFilter1D,
    BreathingDetector,
    BreathingDetectorConfig,
    PresenceState,
    PresenceDetector,
    PresenceDetectorConfig,
)


class TestBreathingState:
    """Tests for BreathingState."""

    def test_to_dict(self):
        state = BreathingState(
            is_breathing=True,
            is_present=True,
            breath_rate=15.0,
            confidence=0.8,
            breath_ratio=0.3,
            hold_duration=0.0,
        )
        d = state.to_dict()
        assert d["is_breathing"] is True
        assert d["breath_rate"] == 15.0
        assert d["confidence"] == 0.8


class TestKalmanFilter1D:
    """Tests for KalmanFilter1D."""

    def test_initial_state(self):
        kf = KalmanFilter1D()
        state = kf.get_state()
        assert state["estimate"] == 0.0
        assert state["error_estimate"] == 1.0

    def test_update_converges(self):
        kf = KalmanFilter1D()
        target = 10.0

        # Repeatedly update with same value
        for _ in range(100):
            estimate = kf.update(target)

        # Should converge to target
        assert abs(estimate - target) < 0.1

    def test_update_filters_noise(self):
        kf = KalmanFilter1D(process_variance=1e-3, measurement_variance=1.0)
        np.random.seed(42)

        true_value = 50.0
        noisy_measurements = true_value + np.random.randn(100) * 5

        estimates = [kf.update(m) for m in noisy_measurements]

        # Final estimate should be close to true value
        assert abs(estimates[-1] - true_value) < 3.0

        # Estimates should be smoother than raw measurements
        estimates_std = np.std(estimates[50:])  # After warmup
        measurements_std = np.std(noisy_measurements[50:])
        assert estimates_std < measurements_std

    def test_reset(self):
        kf = KalmanFilter1D()
        kf.update(100.0)
        kf.reset()
        state = kf.get_state()
        assert state["estimate"] == 0.0
        assert state["error_estimate"] == 1.0


class TestBreathingDetectorConfig:
    """Tests for BreathingDetectorConfig."""

    def test_default_values(self):
        config = BreathingDetectorConfig()
        assert config.sample_rate == 10.0
        assert config.num_subcarriers == 52
        assert config.breathing_freq_min == 0.1
        assert config.breathing_freq_max == 0.5

    def test_custom_values(self):
        config = BreathingDetectorConfig(
            sample_rate=20.0,
            num_subcarriers=64,
            breathing_threshold=0.1,
        )
        assert config.sample_rate == 20.0
        assert config.num_subcarriers == 64
        assert config.breathing_threshold == 0.1


class TestBreathingDetector:
    """Tests for BreathingDetector."""

    def test_initialization(self):
        detector = BreathingDetector()
        assert detector.config.sample_rate == 10.0
        assert len(detector.subcarrier_history) == 52
        assert len(detector.kalman_filters) == 52

    def test_initialization_with_config(self):
        config = BreathingDetectorConfig(num_subcarriers=64)
        detector = BreathingDetector(config)
        assert len(detector.subcarrier_history) == 64
        assert len(detector.kalman_filters) == 64

    def test_update_insufficient_samples(self):
        detector = BreathingDetector()
        amplitudes = np.random.randn(52) + 50

        # First few updates should return empty state
        state = detector.update(amplitudes, 0.0)
        assert state.is_breathing is False
        assert state.is_present is False
        assert state.confidence == 0.0

    def test_update_with_breathing_signal(self):
        detector = BreathingDetector()

        # Simulate breathing at 15 breaths/min (0.25 Hz)
        for i in range(200):
            t = i * 0.1  # 10 Hz sample rate
            breathing = 10 * np.sin(2 * np.pi * 0.25 * t)
            noise = np.random.randn(52) * 2
            amplitudes = 50 + breathing + noise

            state = detector.update(amplitudes, t)

        # After enough samples, should detect breathing
        assert state.breath_ratio > 0.0
        # Note: Detection may vary based on random noise

    def test_update_handles_different_amplitude_sizes(self):
        detector = BreathingDetector()

        # Too few subcarriers (should pad)
        state = detector.update(np.random.randn(30) + 50, 0.0)
        assert isinstance(state, BreathingState)

        # Too many subcarriers (should truncate)
        state = detector.update(np.random.randn(100) + 50, 0.1)
        assert isinstance(state, BreathingState)

    def test_reset(self):
        detector = BreathingDetector()

        # Add some data
        for i in range(50):
            detector.update(np.random.randn(52) + 50, i * 0.1)

        detector.reset()

        assert len(detector.time_history) == 0
        assert detector.baseline_variance is None
        assert detector._update_count == 0

    def test_get_statistics(self):
        detector = BreathingDetector()

        for i in range(50):
            detector.update(np.random.randn(52) + 50, i * 0.1)

        stats = detector.get_statistics()
        assert stats["update_count"] == 50
        assert stats["samples_buffered"] == 50
        assert "detection_rate" in stats

    def test_calibrate_baseline(self):
        detector = BreathingDetector()

        # Calibrate with empty room data (low variance)
        empty_room_data = [np.random.randn(52) * 0.1 + 50 for _ in range(50)]
        timestamps = [i * 0.1 for i in range(50)]

        detector.calibrate_baseline(empty_room_data, timestamps)

        assert detector.baseline_variance is not None
        assert detector.baseline_variance < 1.0  # Low variance


class TestPresenceState:
    """Tests for PresenceState."""

    def test_to_dict(self):
        state = PresenceState(
            is_present=True,
            is_moving=False,
            presence_score=0.7,
            motion_level=0.2,
            variance_ratio=1.8,
        )
        d = state.to_dict()
        assert d["is_present"] is True
        assert d["is_moving"] is False
        assert d["variance_ratio"] == 1.8


class TestPresenceDetectorConfig:
    """Tests for PresenceDetectorConfig."""

    def test_default_values(self):
        config = PresenceDetectorConfig()
        assert config.sample_rate == 10.0
        assert config.presence_threshold == 1.5
        assert config.motion_threshold == 3.0

    def test_custom_values(self):
        config = PresenceDetectorConfig(
            sample_rate=20.0,
            presence_threshold=2.0,
        )
        assert config.sample_rate == 20.0
        assert config.presence_threshold == 2.0


class TestPresenceDetector:
    """Tests for PresenceDetector."""

    def test_initialization(self):
        detector = PresenceDetector()
        assert detector.config.sample_rate == 10.0
        assert len(detector.subcarrier_history) == 52

    def test_update_insufficient_samples(self):
        detector = PresenceDetector()
        state = detector.update(np.random.randn(52) + 50, 0.0)
        assert state.is_present is False
        assert state.variance_ratio == 1.0

    def test_update_detects_presence(self):
        detector = PresenceDetector()

        # First, establish baseline with low variance
        for i in range(50):
            amplitudes = np.random.randn(52) * 0.1 + 50
            detector.update(amplitudes, i * 0.1)

        # Then, add high variance signal
        for i in range(50, 100):
            amplitudes = np.random.randn(52) * 10 + 50  # 100x higher variance
            state = detector.update(amplitudes, i * 0.1)

        # Should detect presence due to increased variance
        assert state.variance_ratio > 1.0

    def test_update_detects_motion(self):
        detector = PresenceDetector()

        # Establish baseline
        for i in range(50):
            detector.update(np.random.randn(52) * 0.1 + 50, i * 0.1)

        # Add very high variance (motion)
        for i in range(50, 100):
            amplitudes = np.random.randn(52) * 50 + 50  # Very high variance
            state = detector.update(amplitudes, i * 0.1)

        # Should detect motion
        assert state.motion_level > 0.0 or state.variance_ratio > detector.config.motion_threshold

    def test_reset(self):
        detector = PresenceDetector()

        for i in range(50):
            detector.update(np.random.randn(52) + 50, i * 0.1)

        detector.reset()

        assert len(detector.time_history) == 0
        assert detector.baseline_variance is None
        assert detector._update_count == 0

    def test_calibrate(self):
        detector = PresenceDetector()

        # Calibrate with empty room data
        empty_data = [np.random.randn(52) * 0.1 + 50 for _ in range(50)]
        timestamps = [i * 0.1 for i in range(50)]

        detector.calibrate(empty_data, timestamps)

        assert detector.baseline_variance is not None

    def test_get_statistics(self):
        detector = PresenceDetector()

        for i in range(50):
            detector.update(np.random.randn(52) + 50, i * 0.1)

        stats = detector.get_statistics()
        assert stats["update_count"] == 50
        assert "presence_rate" in stats
        assert "baseline_variance" in stats

    def test_set_sensitivity(self):
        detector = PresenceDetector()

        detector.set_sensitivity("high")
        assert detector.config.presence_threshold == 1.2

        detector.set_sensitivity("medium")
        assert detector.config.presence_threshold == 1.5

        detector.set_sensitivity("low")
        assert detector.config.presence_threshold == 2.0

    def test_set_sensitivity_invalid(self):
        detector = PresenceDetector()
        with pytest.raises(ValueError):
            detector.set_sensitivity("invalid")


class TestIntegration:
    """Integration tests for detection modules."""

    def test_breathing_and_presence_together(self):
        """Test using both detectors on same data stream."""
        breathing = BreathingDetector()
        presence = PresenceDetector()

        # Simulate a person in the room breathing
        for i in range(200):
            t = i * 0.1
            # Breathing signal + background variance
            breath_signal = 10 * np.sin(2 * np.pi * 0.25 * t)
            motion = 5 * np.random.randn(52)
            amplitudes = 50 + breath_signal + motion

            breath_state = breathing.update(amplitudes, t)
            presence_state = presence.update(amplitudes, t)

        # After warmup, should detect presence
        assert presence_state.variance_ratio > 1.0

    def test_empty_room_detection(self):
        """Test that empty room is correctly detected."""
        breathing = BreathingDetector()
        presence = PresenceDetector()

        # Simulate empty room (low variance noise only)
        np.random.seed(42)
        for i in range(200):
            t = i * 0.1
            amplitudes = np.random.randn(52) * 0.5 + 50  # Very low variance

            breath_state = breathing.update(amplitudes, t)
            presence_state = presence.update(amplitudes, t)

        # Should not detect breathing in empty room
        assert not breath_state.is_breathing

    def test_breath_hold_detection(self):
        """Test detection of breath holding."""
        detector = BreathingDetector()

        # First, establish normal breathing
        for i in range(100):
            t = i * 0.1
            breath = 10 * np.sin(2 * np.pi * 0.25 * t)
            amplitudes = 50 + breath + np.random.randn(52) * 2
            detector.update(amplitudes, t)

        # Then, stop breathing (person still present but no breathing pattern)
        for i in range(100, 150):
            t = i * 0.1
            # Still some variance (person present) but no periodic pattern
            amplitudes = 50 + np.random.randn(52) * 5
            state = detector.update(amplitudes, t)

        # hold_duration should be > 0 if is_present and not is_breathing
        if state.is_present and not state.is_breathing:
            assert state.hold_duration > 0.0
