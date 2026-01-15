"""Tests for CSI data validation framework."""

import numpy as np
import pytest
from wavira.data.validation import (
    CSIDataConfig,
    CSIFormatValidator,
    CSIAnomalyDetector,
    CSIQualityMetrics,
    CSIDuplicateDetector,
    MissingDataHandler,
    CSIDataValidator,
    ValidationLevel,
    ValidationResult,
    ValidationIssue,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_initial_state(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert len(result.metrics) == 0

    def test_add_error_issue(self):
        result = ValidationResult(is_valid=True)
        result.add_issue(ValidationLevel.ERROR, "TEST_ERROR", "Test error message")
        assert result.is_valid is False
        assert result.has_errors() is True
        assert result.error_count() == 1

    def test_add_warning_issue(self):
        result = ValidationResult(is_valid=True)
        result.add_issue(ValidationLevel.WARNING, "TEST_WARNING", "Test warning message")
        assert result.is_valid is True  # Warnings don't invalidate
        assert result.has_warnings() is True
        assert result.warning_count() == 1

    def test_add_info_issue(self):
        result = ValidationResult(is_valid=True)
        result.add_issue(ValidationLevel.INFO, "TEST_INFO", "Test info message")
        assert result.is_valid is True
        assert result.has_errors() is False
        assert result.has_warnings() is False

    def test_issue_with_details(self):
        result = ValidationResult(is_valid=True)
        result.add_issue(
            ValidationLevel.ERROR,
            "TEST_ERROR",
            "Test message",
            location="test.py:10",
            details={"key": "value"}
        )
        assert len(result.issues) == 1
        assert result.issues[0].location == "test.py:10"
        assert result.issues[0].details == {"key": "value"}


class TestCSIFormatValidator:
    """Tests for CSIFormatValidator class."""

    def test_validate_shape_3d(self):
        validator = CSIFormatValidator()
        data = np.zeros((1, 52, 100))
        result = validator.validate_shape(data, expected_dims=3)
        assert result.is_valid is True
        assert result.metrics["ndim"] == 3

    def test_validate_shape_wrong_dims(self):
        validator = CSIFormatValidator()
        data = np.zeros((52, 100))  # 2D instead of 3D
        result = validator.validate_shape(data, expected_dims=3)
        assert result.is_valid is False
        assert any(i.code == "INVALID_DIMENSIONS" for i in result.issues)

    def test_validate_shape_empty(self):
        validator = CSIFormatValidator()
        data = np.zeros((0, 0, 0))
        result = validator.validate_shape(data)
        assert any(i.code == "EMPTY_DATA" for i in result.issues)

    def test_validate_subcarriers_correct(self):
        validator = CSIFormatValidator()
        data = np.zeros((1, 52, 100))
        result = validator.validate_subcarriers(data, subcarrier_axis=1)
        assert result.is_valid is True
        assert result.metrics["n_subcarriers"] == 52

    def test_validate_subcarriers_wrong_count(self):
        validator = CSIFormatValidator()
        data = np.zeros((1, 64, 100))  # 64 instead of 52
        result = validator.validate_subcarriers(data, subcarrier_axis=1)
        assert any(i.code == "UNEXPECTED_SUBCARRIER_COUNT" for i in result.issues)

    def test_validate_packets_correct(self):
        validator = CSIFormatValidator()
        data = np.zeros((1, 52, 100))
        result = validator.validate_packets(data)
        assert result.is_valid is True
        assert result.metrics["n_packets"] == 100

    def test_validate_packets_too_few(self):
        validator = CSIFormatValidator()
        data = np.zeros((1, 52, 5))  # Less than min_packets=10
        result = validator.validate_packets(data)
        assert result.is_valid is False
        assert any(i.code == "TOO_FEW_PACKETS" for i in result.issues)

    def test_validate_packets_too_many(self):
        config = CSIDataConfig(max_packets=100)
        validator = CSIFormatValidator(config)
        data = np.zeros((1, 52, 500))
        result = validator.validate_packets(data)
        assert any(i.code == "TOO_MANY_PACKETS" for i in result.issues)

    def test_validate_format_complete(self):
        validator = CSIFormatValidator()
        data = np.random.randn(1, 52, 100)
        result = validator.validate_format(data)
        assert result.is_valid is True
        assert "shape" in result.metrics
        assert "n_subcarriers" in result.metrics
        assert "n_packets" in result.metrics

    def test_validate_format_non_numeric(self):
        validator = CSIFormatValidator()
        data = np.array([["a", "b"], ["c", "d"]])
        result = validator.validate_format(data)
        assert result.is_valid is False
        assert any(i.code == "INVALID_DTYPE" for i in result.issues)


class TestCSIAnomalyDetector:
    """Tests for CSIAnomalyDetector class."""

    def test_detect_nan_inf_clean(self):
        detector = CSIAnomalyDetector()
        data = np.random.randn(1, 52, 100)
        result = detector.detect_nan_inf(data)
        assert result.is_valid is True
        assert result.metrics["nan_count"] == 0
        assert result.metrics["inf_count"] == 0

    def test_detect_nan_excessive(self):
        detector = CSIAnomalyDetector()
        data = np.random.randn(1, 52, 100)
        # Set 5% to NaN (exceeds default 1% threshold)
        mask = np.random.random(data.shape) < 0.05
        data[mask] = np.nan
        result = detector.detect_nan_inf(data)
        assert result.is_valid is False
        assert any(i.code == "EXCESSIVE_NAN" for i in result.issues)

    def test_detect_inf_excessive(self):
        detector = CSIAnomalyDetector()
        data = np.random.randn(1, 52, 100)
        # Set 1% to Inf (exceeds default 0.1% threshold)
        mask = np.random.random(data.shape) < 0.01
        data[mask] = np.inf
        result = detector.detect_nan_inf(data)
        assert result.is_valid is False
        assert any(i.code == "EXCESSIVE_INF" for i in result.issues)

    def test_detect_zero_values_normal(self):
        detector = CSIAnomalyDetector()
        data = np.random.randn(1, 52, 100) + 10  # Shift away from zero
        result = detector.detect_zero_values(data)
        assert result.is_valid is True

    def test_detect_zero_values_excessive(self):
        detector = CSIAnomalyDetector()
        data = np.zeros((1, 52, 100))  # All zeros
        result = detector.detect_zero_values(data)
        assert any(i.code == "EXCESSIVE_ZEROS" for i in result.issues)

    def test_detect_outliers_clean(self):
        detector = CSIAnomalyDetector()
        data = np.random.randn(1, 52, 100)  # Normal distribution
        result = detector.detect_outliers(data)
        assert result.is_valid is True
        assert "mean" in result.metrics
        assert "std" in result.metrics

    def test_detect_outliers_many(self):
        # Use lower threshold to make detection easier
        config = CSIDataConfig(outlier_std_threshold=2.0)  # Lower threshold
        detector = CSIAnomalyDetector(config)
        # Create tight normal distribution
        np.random.seed(42)
        data = np.random.randn(1, 52, 100) * 0.1  # Small variance
        # Add 10% values that are far from mean
        outlier_count = int(data.size * 0.1)
        outlier_indices = np.random.choice(data.size, size=outlier_count, replace=False)
        flat_data = data.flatten()
        # Set outliers to values that will exceed 2 std threshold
        flat_data[outlier_indices] = 5.0  # Far from 0 mean
        data = flat_data.reshape((1, 52, 100))
        result = detector.detect_outliers(data)
        # Should detect outliers (they're 50x the std)
        assert result.metrics["outlier_count"] > 0
        # With 10% outliers and 2 std threshold, should trigger warning
        assert result.metrics["outlier_ratio"] > 0.05
        assert any(i.code == "HIGH_OUTLIER_RATIO" for i in result.issues)

    def test_detect_amplitude_range_valid(self):
        config = CSIDataConfig(amplitude_min=0, amplitude_max=100)
        detector = CSIAnomalyDetector(config)
        data = np.random.uniform(10, 90, (1, 52, 100))
        result = detector.detect_amplitude_range(data)
        assert result.is_valid is True

    def test_detect_amplitude_range_below(self):
        config = CSIDataConfig(amplitude_min=0, amplitude_max=100)
        detector = CSIAnomalyDetector(config)
        data = np.random.uniform(-50, 50, (1, 52, 100))  # Some below 0
        result = detector.detect_amplitude_range(data)
        assert any(i.code == "AMPLITUDE_BELOW_RANGE" for i in result.issues)

    def test_detect_amplitude_range_above(self):
        config = CSIDataConfig(amplitude_min=0, amplitude_max=100)
        detector = CSIAnomalyDetector(config)
        data = np.random.uniform(50, 150, (1, 52, 100))  # Some above 100
        result = detector.detect_amplitude_range(data)
        assert any(i.code == "AMPLITUDE_ABOVE_RANGE" for i in result.issues)

    def test_detect_all_anomalies(self):
        detector = CSIAnomalyDetector()
        data = np.random.randn(1, 52, 100)
        result = detector.detect_all_anomalies(data)
        assert "nan_count" in result.metrics
        assert "zero_count" in result.metrics
        assert "mean" in result.metrics


class TestCSIQualityMetrics:
    """Tests for CSIQualityMetrics class."""

    def test_signal_to_noise_ratio_high_snr(self):
        metrics = CSIQualityMetrics()
        # Clean signal with little noise
        t = np.linspace(0, 1, 100)
        signal = 10 * np.sin(2 * np.pi * 5 * t)
        data = signal + np.random.randn(100) * 0.1  # Low noise
        snr = metrics.signal_to_noise_ratio(data)
        assert snr > 10  # High SNR expected

    def test_signal_to_noise_ratio_low_snr(self):
        metrics = CSIQualityMetrics()
        # Mostly noise
        data = np.random.randn(100) * 10
        snr = metrics.signal_to_noise_ratio(data)
        assert snr < 10  # Low SNR expected

    def test_signal_to_noise_ratio_empty(self):
        metrics = CSIQualityMetrics()
        data = np.array([])
        snr = metrics.signal_to_noise_ratio(data)
        assert snr == 0.0

    def test_temporal_consistency_smooth(self):
        metrics = CSIQualityMetrics()
        # Smooth signal
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * t)
        consistency = metrics.temporal_consistency(data)
        assert consistency < 0.5  # Smooth = low score

    def test_temporal_consistency_noisy(self):
        metrics = CSIQualityMetrics()
        # Random noise
        data = np.random.randn(100)
        consistency = metrics.temporal_consistency(data)
        assert consistency > 0.1  # Noisy = higher score

    def test_subcarrier_correlation_high(self):
        metrics = CSIQualityMetrics()
        # All subcarriers have same pattern
        pattern = np.sin(np.linspace(0, 2 * np.pi, 100))
        data = np.tile(pattern, (52, 1)) + np.random.randn(52, 100) * 0.1
        data = data.reshape(1, 52, 100)
        corr = metrics.subcarrier_correlation(data)
        assert corr > 0.5  # High correlation expected

    def test_subcarrier_correlation_low(self):
        metrics = CSIQualityMetrics()
        # Random uncorrelated subcarriers
        data = np.random.randn(1, 52, 100)
        corr = metrics.subcarrier_correlation(data)
        assert corr < 0.5  # Low correlation expected

    def test_compute_all_metrics(self):
        metrics = CSIQualityMetrics()
        data = np.random.randn(1, 52, 100)
        all_metrics = metrics.compute_all_metrics(data)
        assert "snr_db" in all_metrics
        assert "temporal_consistency" in all_metrics
        assert "subcarrier_correlation" in all_metrics


class TestCSIDuplicateDetector:
    """Tests for CSIDuplicateDetector class."""

    def test_compute_hash_same(self):
        detector = CSIDuplicateDetector()
        data = np.array([1.0, 2.0, 3.0])
        hash1 = detector.compute_hash(data)
        hash2 = detector.compute_hash(data.copy())
        assert hash1 == hash2

    def test_compute_hash_different(self):
        detector = CSIDuplicateDetector()
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([1.0, 2.0, 4.0])
        hash1 = detector.compute_hash(data1)
        hash2 = detector.compute_hash(data2)
        assert hash1 != hash2

    def test_find_duplicates_none(self):
        detector = CSIDuplicateDetector()
        samples = [np.random.randn(10) for _ in range(5)]
        duplicates = detector.find_duplicates(samples)
        assert len(duplicates) == 0

    def test_find_duplicates_exact(self):
        detector = CSIDuplicateDetector()
        sample = np.random.randn(10)
        samples = [sample.copy() for _ in range(3)]  # 3 identical
        duplicates = detector.find_duplicates(samples)
        assert len(duplicates) == 2  # (0,1) and (0,2)

    def test_find_near_duplicates(self):
        detector = CSIDuplicateDetector()
        base = np.random.randn(100)
        samples = [
            base,
            base + np.random.randn(100) * 0.001,  # Near duplicate
            np.random.randn(100),  # Different
        ]
        near_dups = detector.find_near_duplicates(samples, similarity_threshold=0.99)
        assert len(near_dups) == 1
        assert near_dups[0][0] == 0
        assert near_dups[0][1] == 1
        assert near_dups[0][2] > 0.99

    def test_detect_duplicates(self):
        detector = CSIDuplicateDetector()
        sample = np.random.randn(10)
        samples = [sample.copy(), sample.copy(), np.random.randn(10)]
        result = detector.detect_duplicates(samples)
        assert result.metrics["exact_duplicates"] == 1
        assert any(i.code == "EXACT_DUPLICATES_FOUND" for i in result.issues)


class TestMissingDataHandler:
    """Tests for MissingDataHandler class."""

    def test_detect_missing_none(self):
        handler = MissingDataHandler()
        data = np.random.randn(10, 10)
        mask, ratio = handler.detect_missing(data)
        assert ratio == 0.0
        assert np.sum(mask) == 0

    def test_detect_missing_some(self):
        handler = MissingDataHandler()
        data = np.random.randn(10, 10)
        data[0, 0] = np.nan
        data[5, 5] = np.nan
        mask, ratio = handler.detect_missing(data)
        assert np.sum(mask) == 2
        assert ratio == pytest.approx(0.02)

    def test_interpolate_missing_linear(self):
        handler = MissingDataHandler()
        data = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])
        result = handler.interpolate_missing(data, axis=-1, method="linear")
        expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpolate_missing_nearest(self):
        handler = MissingDataHandler()
        data = np.array([[1.0, np.nan, np.nan, 5.0]])
        result = handler.interpolate_missing(data, axis=-1, method="nearest")
        assert result[0, 1] == 1.0  # Nearest to index 0
        assert result[0, 2] == 5.0  # Nearest to index 3

    def test_interpolate_missing_all_nan(self):
        handler = MissingDataHandler()
        data = np.array([[np.nan, np.nan, np.nan]])
        result = handler.interpolate_missing(data, axis=-1)
        np.testing.assert_array_equal(result, [[0.0, 0.0, 0.0]])

    def test_drop_missing_samples(self):
        handler = MissingDataHandler()
        samples = [
            np.array([1.0, 2.0, 3.0]),  # No missing
            np.array([1.0, np.nan, 3.0]),  # 33% missing
            np.array([np.nan, np.nan, 3.0]),  # 67% missing
            np.array([1.0, 2.0, np.nan]),  # 33% missing
        ]
        kept, indices = handler.drop_missing_samples(samples, threshold=0.4)
        assert len(kept) == 3
        assert indices == [0, 1, 3]


class TestCSIDataValidator:
    """Tests for main CSIDataValidator class."""

    def test_validate_valid_data(self):
        validator = CSIDataValidator()
        data = np.random.randn(1, 52, 100).astype(np.float32)
        result = validator.validate(data)
        assert result.is_valid is True

    def test_validate_invalid_shape(self):
        validator = CSIDataValidator()
        data = np.random.randn(52, 100)  # 2D instead of 3D
        result = validator.validate(data)
        assert result.is_valid is False

    def test_validate_with_nan(self):
        config = CSIDataConfig(nan_threshold=0.001)
        validator = CSIDataValidator(config)
        data = np.random.randn(1, 52, 100)
        data[0, :5, :5] = np.nan  # 5*5=25 NaN values out of 5200
        result = validator.validate(data)
        # NaN ratio is 25/5200 = 0.48%, exceeds 0.1% threshold
        assert any(i.code == "EXCESSIVE_NAN" for i in result.issues)

    def test_validate_skip_format(self):
        validator = CSIDataValidator()
        data = np.random.randn(52, 100)  # Would fail format check
        result = validator.validate(data, check_format=False)
        # Should proceed without format errors
        assert "format" not in result.metrics

    def test_validate_skip_anomalies(self):
        validator = CSIDataValidator()
        data = np.random.randn(1, 52, 100)
        result = validator.validate(data, check_anomalies=False)
        assert "anomalies" not in result.metrics

    def test_validate_skip_quality(self):
        validator = CSIDataValidator()
        data = np.random.randn(1, 52, 100)
        result = validator.validate(data, check_quality=False)
        assert "quality" not in result.metrics

    def test_validate_dataset_empty(self):
        validator = CSIDataValidator()
        result = validator.validate_dataset([])
        assert result.is_valid is False
        assert any(i.code == "EMPTY_DATASET" for i in result.issues)

    def test_validate_dataset_valid(self):
        validator = CSIDataValidator()
        samples = [np.random.randn(1, 52, 100) for _ in range(5)]
        result = validator.validate_dataset(samples)
        assert result.is_valid is True
        assert result.metrics["n_samples"] == 5

    def test_validate_dataset_with_invalid(self):
        validator = CSIDataValidator()
        samples = [
            np.random.randn(1, 52, 100),
            np.random.randn(52, 100),  # Invalid shape
            np.random.randn(1, 52, 100),
        ]
        result = validator.validate_dataset(samples, check_duplicates=False)
        assert result.metrics["invalid_samples"] == 1

    def test_validate_dataset_with_duplicates(self):
        validator = CSIDataValidator()
        sample = np.random.randn(1, 52, 100)
        samples = [sample.copy() for _ in range(3)]
        result = validator.validate_dataset(samples, check_duplicates=True)
        assert result.metrics["duplicates"]["exact"] == 2

    def test_generate_report(self):
        validator = CSIDataValidator()
        data = np.random.randn(1, 52, 100)
        result = validator.validate(data)
        report = validator.generate_report(result)
        assert "CSI Data Validation Report" in report
        assert "PASSED" in report or "FAILED" in report
        assert "Metrics:" in report


class TestCSIDataConfig:
    """Tests for CSIDataConfig class."""

    def test_default_values(self):
        config = CSIDataConfig()
        assert config.expected_subcarriers == 52
        assert config.expected_rx_antennas == 1
        assert config.min_packets == 10
        assert config.max_packets == 10000

    def test_custom_values(self):
        config = CSIDataConfig(
            expected_subcarriers=64,
            min_packets=50,
            nan_threshold=0.05
        )
        assert config.expected_subcarriers == 64
        assert config.min_packets == 50
        assert config.nan_threshold == 0.05


class TestValidationIntegration:
    """Integration tests for the validation framework."""

    def test_full_validation_pipeline(self):
        validator = CSIDataValidator()

        # Create realistic CSI data
        n_antennas = 1
        n_subcarriers = 52
        n_packets = 200

        # Simulate amplitude data with some noise
        t = np.linspace(0, 2 * np.pi, n_packets)
        base_signal = 50 + 10 * np.sin(t)  # Breathing-like pattern

        data = np.zeros((n_antennas, n_subcarriers, n_packets))
        for i in range(n_subcarriers):
            data[0, i, :] = base_signal + np.random.randn(n_packets) * 2

        result = validator.validate(data)

        assert result.is_valid is True
        assert "format" in result.metrics
        assert "anomalies" in result.metrics
        assert "quality" in result.metrics

        # Check quality metrics are reasonable
        quality = result.metrics["quality"]
        assert quality["snr_db"] > 0  # Should have positive SNR
        assert quality["temporal_consistency"] < 1.0
        assert quality["subcarrier_correlation"] > 0.5  # Similar patterns

    def test_validation_with_real_issues(self):
        validator = CSIDataValidator()

        # Create data with multiple issues
        data = np.random.randn(1, 52, 100)

        # Add some NaN values
        data[0, 0:2, 0:5] = np.nan

        # Add some extreme outliers
        data[0, 25, 50] = 1000

        result = validator.validate(data)

        # Should still be valid (NaN under threshold)
        # but have some warnings
        report = validator.generate_report(result)
        assert "Metrics:" in report

    def test_dataset_validation_pipeline(self):
        validator = CSIDataValidator()

        # Create a dataset with mixed quality
        samples = []
        for i in range(10):
            sample = np.random.randn(1, 52, 100)
            samples.append(sample)

        # Add one duplicate
        samples.append(samples[0].copy())

        # Add one invalid sample
        samples.append(np.random.randn(52, 100))  # Wrong shape

        result = validator.validate_dataset(samples)

        assert result.metrics["n_samples"] == 12
        assert result.metrics["invalid_samples"] == 1
        assert result.metrics["duplicates"]["exact"] == 1
